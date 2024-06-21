# MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models

This repository is the implementation of the paper "MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models".

## Directory structure

- `base_model`: MeteoRA model
- `ckpt`: the datasets and dataset processing code.
- `eval`: the evaluation results and evaluation code.
- `MoELoRA`: MeteoRA module and adapted PEFT code.

## Usage

### Preparation

1. Install necessary packages:
```shell
pip install -r requirements.txt
```

2. Prepare the datasets. MeteoRA requires datasets in JSONL format. In our paper, MeteoRA involves 28 tasks which are almost selected from BIGBench dataset. The format in BIGBench is JSON. So we need to run the following scripts to generate the datasets for all tasks in JSONL format.
```shell
cd data
python create_dataset.py --task all
```

If you just want to create a specific dataset, run:
```shell
cd data
python create_dataset.py --task <task_name>

```
3. Prepare *composite-n* tasks. Please refer to our paper for the defition of *composite-n* task.
```shell
python create_composite.py --n <n>
```
We prepared `n=3`, `n=5` and `n=10` few-shot dataset generating code. Before generating, please ensure that the sub-tasks to composite *composite-n* task have been included in `data/datasets`.


4. Prepare LoRA adapters checkpoint and MeteoRA model checkpoint. You can train by yourself or download ours([LlaMA2](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama2_13b) and [LlaMA3](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama3_8b) as base model) by:
```shell
python download_ckpt.py
```

5. Change file path in `configs/config.yaml`.
For example, the paths in config.yaml for MeteoRA are as follows:

```sh
base_model_path: 'meta-llama3/Meta-Llama-3-8B'
meteora_ckpt_path: 'ckpt/llama3_8b/llama3_8b_meteora/top_2'
adapter_dir: 'ckpt/llama3_8b/llama3_8b_peft'
```

### Evaluation

Running a benchmark with MeteoRA model:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> 
```

For example:
```shell
python eval_model.py --task composite_10 --batch_size 4 
```

**Note:** If you want to run a *composite-n* task, please set a larger *temperature* value (`self.T` in `MoELoRA/layer.py`). As a reference, `15`, `20` and `30` for `n=3`, `n=5` and `n=10`. For evaluation on each single task, we set `self.T` to be the default value (`self.T=1`) in our experiments.


If you want to save the evaluation result, execute the following command:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --save
```

Debug mode (model output and ground truth will be shown in the console):
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --debug
```

Running a benchmark with PEFT model:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --model <adapter_name>
```

### Train MeteoRA model

0. Prepare LoRA adapters and the corresponding datasets in JSONL format. One should prepare the JSONL dataset for each LoRA adapter. Please refer to the above dataest preparation step and the script ```data/create_dataset.py```. All LoRA adapters and all datasets should be put in the adpater and dataset folders with the same subfolder name, repsectively. For example, the folder structure should be as follows:
```
- lora_adapters
      - adapter_name1
      - adapter_name2
      - ...
- datasets
      - dataset_name1
      - dataset_name2
      - ...
```

1. Change file path in `run_meteora_train_fsdp.sh`. 


2. Train MeteoRA model:
```shell
sh run_meteora_train_fsdp.sh
```

Note: the triton acceleration only support the inference mode in the current version. Thus, one should use the following settings when training MeteoRA model:

```sh
export MOELINEAR_USE_ACCELERATE_FWD=0

export MOELINEAR_FWD_INNER_LOOP_MODE='batch'

export MOELINEAR_ACCELERATE_FWD_BACKEND='torch'
export MOELINEAR_ACCELERATE_FWD_BACKEND_TORCH_VERSION='v1'
```

## Citation

If you use MeteoRA for your research, please cite our [paper](https://arxiv.org/abs/2405.13053):
```bibtex
@misc{xu2024meteora,
      title={MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models}, 
      author={Jingwei Xu and Junyu Lai and Yunpeng Huang},
      year={2024},
      eprint={2405.13053},
      archivePrefix={arXiv},
}
```