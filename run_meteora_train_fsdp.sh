export MOELINEAR_USE_ACCELERATE_FWD=0
# MOELINEAR_FWD_INNER_LOOP_MODE='all'

export MOELINEAR_FWD_INNER_LOOP_MODE='batch'

export MOELINEAR_ACCELERATE_FWD_BACKEND='torch'
export MOELINEAR_ACCELERATE_FWD_BACKEND_TORCH_VERSION='v1'

# export MOELINEAR_ACCELERATE_FWD_BACKEND='triton'
# export MOELINEAR_ACCELERATE_FWD_BACKEND_TRITON_VERSION='v4'

accelerate launch --num_processes=4 --config_file "configs/fsdp_config.yaml" meteora_train.py \
--model_name "/data1/model/llama3/meta-llama3/Meta-Llama-3-8B" \
--tasks_datasets_prefix "/data0/ljy/workspace/BIG-bench/fuze_28_no_sys/" \
--lora_path_prefix "/data0/ljy/workspace/LLaMA-Factory/ckpt/llama3_8b_fuze27_no_sys/" \
--default_task "alpaca" \
--max_seq_len 4096 \
--max_steps 15000 \
--logging_steps 1 \
--eval_steps 2 \
--save_steps 3000 \
--bf16 True \
--packing True \
--output_dir "/data2/xjw/llama-meteor-data/train_gate_and_loras_v2" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 8 \
--dataset_text_field "text" \
--learning_rate 5e-4  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.005 \
--use_flash_attn True \
--use_gradient_checkpointing       