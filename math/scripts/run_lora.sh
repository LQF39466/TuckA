BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/metamath-LoRA-r32-1e-3"
DATA_PATH="pissa-dataset"

batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 32
deepspeed --master_port=16974 --include=localhost:4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj" \
    --lora_rank 32 \
    --lora_dropout 0 \
    --use_dora \
    --data_path $DATA_PATH \
    --sub_task metamath:100000 \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response_torch.jsonl --merged True --batch_size 8
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response_torch.jsonl
