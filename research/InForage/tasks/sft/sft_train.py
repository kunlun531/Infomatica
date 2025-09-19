import os
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
import json
from typing import Dict, List, Optional
import numpy as np
import shutil
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from IPython import embed
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
        # Keep track of best 3 checkpoints based on eval loss
        current_checkpoint = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        
        # Save model state without optimizer states
        self.save_model(current_checkpoint)
        
        checkpoint_info = {
            'loss': eval_loss,
            'checkpoint': current_checkpoint
        }
        
        self.best_eval_losses.append(checkpoint_info)
        self.best_eval_losses.sort(key=lambda x: x['loss'])
        
        # Keep only top 3
        if len(self.best_eval_losses) > 3:
            # Remove checkpoint files for the worst performing one
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

_prompt = """You are an intelligent agent designed to solve complex queries or tasks by retrieving external knowledge and reasoning step by step.

Please follow these instructions carefully:

1. For each piece of information you receive:
   - Think step by step and explain your reasoning inside <think> and </think> tags.

2. If you need more information to proceed:
   - Issue a search by writing your subquery inside <search> and </search> tags.
   - Retrieved results will appear between <evidence> and </evidence> tags.
   - You can conduct multiple searches as needed.

3. When you have collected enough information:
   - Provide your final answer using the <answer> and </answer> tags.
   - Do not include explanations or reasoning in the answer block.
   - Keep your answer concise.

Now, solve the following task:

Task: {task}
"""
class SftCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.pad_token_id = (tokenizer.pad_token_id 
                            if tokenizer.pad_token_id is not None 
                            else tokenizer.eos_token_id)
        self.max_length = max_length
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant")
        self.sft_start_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
        self.sft_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    
    def __call__(self, batch):
        # Transform the batch first
        encodings, batch_chat_text = self.transform_batch_and_tokenize(batch)

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = self.get_sft_training_labels(input_ids, attention_mask, batch_chat_text)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def transform_batch_and_tokenize(self, batch):
        batch_chat_text = []
        for sample in batch:
            conv = sample['conversation']
            messages = []

            for turn in conv:
                if turn['role'] == 'user':
                    content = _prompt.format(task=turn['content'])
                    messages.append({
                        "role": turn['role'],
                        "content": str(content)
                    })
                else:
                    content = turn['content']
                    messages.append({
                        "role": turn['role'],
                        "content": str(content)
                    })
    
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Changed to False since we'll tokenize later
                add_generation_prompt=False
            )

            batch_chat_text.append(chat_text)
        
        encodings = self.tokenizer(batch_chat_text, 
                                 max_length=self.max_length, 
                                 padding="max_length",
                                 truncation=True, 
                                 return_tensors="pt")
        
        return encodings, batch_chat_text


    def get_sft_training_labels(self, input_ids, attention_mask, batch_chat_text):
        n_sample = len(input_ids)
        labels = input_ids.detach().clone()
        ignore_idx = -100
        labels[attention_mask == 0] = ignore_idx
        
        # mask the prompt part
        for i in range(n_sample):
            # find all of intervals satisfying [sft_start_token, ..., sft_end_token]
            chat_text = batch_chat_text[i]
            input_str, target_str = chat_text.split("<|im_start|>assistant") #TODO hard coding for qwen2.5
            target_str = "<|im_start|>assistant" + target_str
            target_str_length = len(self.tokenizer.encode(target_str, add_special_tokens=False))
            # set the postions that not in the intervals to ignore_idx
            labels[i, :-target_str_length] = ignore_idx

        return labels
        
        
class SftDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
        

def main():
    # Set random seed for reproducibility
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(42)
    version = model_args.version
    training_args.output_dir = os.path.join(training_args.output_dir, version)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # Initialize model and tokenizer
    model_name = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    collator = SftCollator(tokenizer, max_length=1024)
    # Load and process dataset
    data = load_jsonl("~/InForage/dataset/sft/train.jsonl")  # Change this to your data path

    # Split into train and eval
    eval_size = 200
    eval_data = data[:eval_size]
    train_data = data[eval_size:]
    
    # Convert to HF datasets
    train_dataset = SftDataset(train_data)
    eval_dataset = SftDataset(eval_data)

    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model state without optimizer states
    trainer.save_model(training_args.output_dir)
    
    # Print information about best checkpoints
    print("\nBest checkpoints based on eval loss:")
    for idx, checkpoint in enumerate(trainer.best_eval_losses):
        print(f"{idx+1}. Loss: {checkpoint['loss']:.4f}, Path: {checkpoint['checkpoint']}")

if __name__ == "__main__":
    # Setup multi-GPU training
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    main()


