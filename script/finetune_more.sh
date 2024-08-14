export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

output_model="../save/MoRE"

mkdir ${output_model}
deepspeed --include localhost:0,1 --master_port 17620 ../finetune.py \
    --model_name_or_path 't5-base' \
    --tasks cola mnli mrpc qnli qqp rte sst2 stsb \
    --max_length 128 \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q k v o wi wo \
    --output_dir ${output_model} \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10000 \
    --epochs 5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --logging_dir ${output_model}/logs \
    --logging_steps 100 \
    --load_best_model_at_end True \
    --dataloader_num_workers 12 \
    --seed 2023 \
    | tee ${output_model}/train.log
