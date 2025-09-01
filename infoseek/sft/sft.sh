export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY="YOUR_WANDB_KEY"
wandb login

BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
TRAIN_DATA_PATH=your/data/pth
RUN_NAME=InfoSeek-Qwen2.5-3B-RFT


deepspeed --num_gpus=8 \
    sft_train_mask_qwen.py \
    --model_name_or_path=${BASE_MODEL} \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --warmup_ratio=0.1 \
    --logging_steps=1 \
    --max_length=10000 \
    --save_only_model=true \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --per_device_eval_batch_size=1 \
    --num_train_epochs=3 \
    --save_strategy='epoch' \
    --eval_strategy='epoch' \
    --save_total_limit=3 \
    --remove_unused_columns=False \
    --log_level="info" \
    --report_to="wandb" \
    --run_name=${RUN_NAME} \
    --version=${RUN_NAME} \
    --train_data_path=${TRAIN_DATA_PATH} \
    --output_dir="./" \
    --resume_from_checkpoint=true \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed="ds_config/ds_config2_new.json" \
    2>&1 | tee ./logs/${RUN_NAME}.log

