# MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models

This is the implementation of the paper "MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models".

## Directory structure

- `base_model`: MeteoRA model
- `ckpt`: the datasets and dataset processing code.
- `eval`: the evaluation results and evaluation code.
- `MoELoRA`: MeteoRA module and adapted PEFT code.

## Usage

### Preparation

1. Install necessary packages:
```
pip install -r requirements.txt
```
2. Prepare datasets:
```
cd data
python create_dataset.py --task <task_name>
```
3. Prepare *composite-n* tasks:
```
python create_composite.py --n <n>
```
We prepared `n=3`, `n=5` and `n=10` few-shot dataset generating code. Before generating, please ensure that the sub-tasks to composite *composite-n* task have been included in `data/datasets`.

4. Prepare LoRA adapters checkpoint and MeteoRA model checkpoint. You can download ours([LlaMA2](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama2_13b) and [LlaMA3](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama3_8b) as base model) or train by yourself.
5. Change file path in `eval_model.py` and `data/create_dataset.py`,  if necessary.

### Evaluation

Running a benchmark with MeteoRA model:
```
python eval_model.py --task <task_name> --batch_size <batch_size> 
```

For example:
```
python eval_model.py --task composite_10 --batch_size 4 
```

Save the evaluation result:
```
python eval_model.py --task <task_name> --batch_size <batch_size> --save
```

Debug mode (model output and ground truth will be shown in the console):
```
python eval_model.py --task <task_name> --batch_size <batch_size> --debug
```

Running a benchmark with PEFT model:
```
python eval_model.py --task <task_name> --batch_size <batch_size> --model <adapter_name>
```

### Train MeteoRA model

1. Change file path in `run_meteora_train_fsdp.sh` and `meteora_train.py`.

2. Train MeteoRA model:
```
sh run_meteora_train_fsdp.sh
```
