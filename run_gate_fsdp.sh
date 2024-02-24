accelerate launch --num_processes=4 --config_file "configs/fsdp_config.yaml" train.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--datasets_names "/data1/dataset/math/GSM8k/main,/data1/dataset/structure/nl2sql/SQL-CTX,GEM/viggo" \
--max_seq_len 4096 \
--max_steps 4000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "full-finetune-llama-chat-asst" \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 8 \
--dataset_text_field "prompt" \
--use_gradient_checkpointing \
--learning_rate 5e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--use_flash_attn True

# --dataset_name "smangrul/code-chat-assistant-v1" \