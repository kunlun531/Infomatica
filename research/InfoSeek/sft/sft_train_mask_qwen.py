import os, ipdb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from torch.utils.data import Dataset, DataLoader
import json, ipdb, random
from typing import Dict, List, Optional
import numpy as np
import shutil
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from IPython import embed
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    version: str = field(
        default="qwen2.5-3b-sft",
        metadata={"help": "The version of the model."}
    )

@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path to the training data."}
    )
    train_data_ratios_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file that contains the ratios of the training data."}
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "The maximum length of the input text."}
    )

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_losses = []
        self.best_checkpoints = []
        
    def save_best_checkpoints(self, eval_loss):
        current_checkpoint = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        self.save_model(current_checkpoint)
        checkpoint_info = {'loss': eval_loss, 'checkpoint': current_checkpoint}
        self.best_eval_losses.append(checkpoint_info)
        self.best_eval_losses.sort(key=lambda x: x['loss'])
        if len(self.best_eval_losses) > 3:
            worst_checkpoint = self.best_eval_losses.pop()
            if os.path.exists(worst_checkpoint['checkpoint']):
                shutil.rmtree(worst_checkpoint['checkpoint'])
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss:
            self.save_best_checkpoints(eval_loss)
        return super().on_evaluate(args, state, control, metrics, **kwargs)

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class SftCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_start_tag = "<information>"
        self.ignore_end_tag = "</information>"

        self.ignore_pattern = re.compile(
            re.escape(self.ignore_start_tag) + r'.*?' + re.escape(self.ignore_end_tag),
            re.DOTALL
        )

    def __call__(self, batch):
        encodings, batch_chat_text = self.transform_batch_and_tokenize(batch)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = self.get_sft_training_labels(input_ids, attention_mask, batch_chat_text)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def transform_batch_and_tokenize(self, batch):
        batch_chat_text = []
        for sample in batch:
            messages = [{"role": turn['role'], "content": str(turn['content'])} for turn in sample['conversation']]
            chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            batch_chat_text.append(chat_text)
        
        return self.tokenizer(
            batch_chat_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        ), batch_chat_text

    def get_sft_training_labels(self, input_ids, attention_mask, batch_chat_text):
        n_sample = len(input_ids)
        labels = input_ids.clone()
        ignore_idx = -100
        labels[attention_mask == 0] = ignore_idx
        
        for i in range(n_sample):
            chat_text = batch_chat_text[i]
            
            try:
                # NOTE: Qwen2.5
                prompt_str, response_str = chat_text.rsplit("<|im_start|>assistant", 1)
                response_str = "<|im_start|>assistant" + response_str
            except ValueError:
                labels[i, :] = ignore_idx
                continue
            
            prompt_token_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
            labels[i, :prompt_token_len] = ignore_idx

            for match in self.ignore_pattern.finditer(response_str):
                prefix_in_response = response_str[:match.start()]

                matched_block = match.group(0)
                prefix_token_len = len(self.tokenizer.encode(prefix_in_response, add_special_tokens=False))
                block_token_len = len(self.tokenizer.encode(matched_block, add_special_tokens=False))

                mask_start_token_idx = prompt_token_len + prefix_token_len
                mask_end_token_idx = mask_start_token_idx + block_token_len
                labels[i, mask_start_token_idx:mask_end_token_idx] = ignore_idx
        # ipdb.set_trace()
        return labels

        
class SftDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
        

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"model_args: \n{model_args}\ndata_args: \n{data_args}\ntraining_args: \n{training_args}\n")
    set_seed(42)
    version = model_args.version
    training_args.output_dir = os.path.join(training_args.output_dir, version)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    model_name = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    tokenizer.padding_side = "right" 
    collator = SftCollator(tokenizer, max_length=data_args.max_length)
    data = load_jsonl(data_args.train_data_path)
    print(f"\nData Example: \n{data[0]}\n")

    eval_size = 200
    eval_data = data[:eval_size]
    train_data = data[eval_size:]
    random.shuffle(train_data)
    
    train_dataset = SftDataset(train_data)
    eval_dataset = SftDataset(eval_data)

    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\nBest checkpoints based on eval loss:")
    for idx, checkpoint in enumerate(trainer.best_eval_losses):
        print(f"{idx+1}. Loss: {checkpoint['loss']:.4f}, Path: {checkpoint['checkpoint']}")

if __name__ == "__main__":
    main()