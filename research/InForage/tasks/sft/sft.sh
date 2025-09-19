export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_PROJECT="sft_search_llm"

export WANDB_API_KEY=123
wandb login
# gradient_accumulation_steps = 8, 4096 * 4 * 8 * 8 = 2097152
deepspeed --num_gpus=8 \
sft_train.py \
--model_name_or_path="~/InForage/checkpoints/qwen-2.5-3B-instruct" \
--weight_decay=0.1 \
--max_grad_norm=1.0 \
--warmup_ratio=0.1 \
--logging_steps=1 \
--max_length=2048 \
--save_only_model=true \
--num_train_epochs=3 \
--save_strategy='epoch' \
--eval_strategy='epoch' \
--save_total_limit=3 \
--remove_unused_columns=False \
--log_level="info" \
--report_to="wandb" \
--run_name="qwen2.5-3b-sft" \  
--version="qwen2.5-3b-sft" \
--output_dir="~/InForage/checkpoints/qwen-2.5-3b-sft" \
--resume_from_checkpoint=true \
--bf16 \
--gradient_checkpointing \
--deepspeed="~/InForage/tasks/sft/ds_config3.json" \
