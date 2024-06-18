name, aux_coef, moe_gate_coef, moe_gate_layer_mse_coef, moe_gate_logits_mse_coef, with token loss, lr, tasks, steps, weight_path, model_name
wandb329, 0.5, 1, 0.01, NA, True, 3e-7, 15, 7000
wandb330, 1.0, 1, 0.001, NA, False, 3e-7, 15, 7000
wandb332, 1.0, 1, 0.001, NA, False, 3e-7, 18, 7000
wandb333, 2.0, 1, 0.001, NA, False, 3e-7, 18, 5000
wandb334, 10.0, 1, 0.001, NA, False, 3e-7, 18, 5000
wandb335, 50.0, 1, 0.001, NA, True, 3e-7, 18, 5000, llama2-13b
wandb337, 50.0, 1, 0.001, NA, True, 3e-7, 18, 5000, llama2-13b
wandb339, 50.0, 1, 0.001, NA, True, 3e-7, 21, 5000, llama2-13b
wandb340, 0.5, 1, 0.001, NA, True, 3e-7, 21, 5000, llama2-13b
wandb342, NA, NA, NA, NA, True, 3e-5, 21, 5000, only token loss , llama2-13b
wandb342, NA, NA, NA, NA, True, 5e-5, 21, 5000, only token loss, llama2-13b
wandb356, 50.0, 1, 0.001, NA, True, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/28tasks-balance-1k-top2-dropout01-layerMSE-withTranslationTasks, llama2-13b
wandb357, 50.0, 1, NA, NA, True, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/28tasks-balance-1k-top2-dropout01-withTranslationTasks, llama2-13b
wandb358, 50.0, 1, NA, NA, True, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/28tasks-balance-1k-top2-dropout01-withTranslationTasks-2round, llama2-13b
wandb359, 50.0, 1, NA, NA, True, 3e-7, 26, 5000, /data2/xjw/llama-meteor-data/27tasks-balance-1k-top2-dropout01-withTranslationBalance, llama2-13b
wandb372, 50.0, 1, NA, NA, True, 3e-7, 26, 5000, /data2/xjw/llama-meteor-data/llama3-8b-27tasks, llama3-8b
wandb373, 50.0, 1, NA, NA, True, 3e-7, 26, 5000, /data2/xjw/llama-meteor-data/llama3-8b-27tasks-2round, llama3-8b
wandb374, 50.0, 1, NA, NA, True, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/llama3-8b-28tasks, llama3-8b
wandb376, 0.5, 1, NA, NA, True, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/llama3-8b-28tasks-5e-1gate, llama3-8b
wandb377, 50, 1, NA, NA, False, 3e-7, 27, 5000, /data2/xjw/llama-meteor-data/llama3-8b-28tasks-top1, llama3-8b
wandb378, 50, 1, NA, NA, True, 3e-7, 28, 5000, /data2/xjw/llama-meteor-data/llama3-8b-29tasks-top2-cmath, llama3-8b

