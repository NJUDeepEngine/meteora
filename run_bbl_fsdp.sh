accelerate launch --num_processes=2 --config_file "configs/fsdp_config.yaml" train_old.py \
--model_name "meta-llama/Llama-2-13b" \
--datasets_names "/data1/dataset/math/GSM8k/main,/data1/dataset/structure/nl2sql/SQL-CTX,GEM/viggo" \
--max_seq_len 4096 \
--max_steps 5000 \
--logging_steps 1 \
--eval_steps 500 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "/data2/xjw/llama-meteor-data/llama3-8b-27tasks" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 8 \
--dataset_text_field "text" \
--use_gradient_checkpointing \
--learning_rate 7e-3  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.05 \
--use_flash_attn True

# --dataset_name "smangrul/code-chat-assistant-v1" \