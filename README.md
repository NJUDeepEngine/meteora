# MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models

This repository contains the implementation of the paper "MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models".

## Directory structure

- `base_model`: Contains the MeteoRA model.
- `ckpt`: Contains the datasets and dataset processing code.
- `eval`: Contains the evaluation results and evaluation code.
- `MoELoRA`: Contains the MeteoRA module and adapted PEFT code.

## Usage

### Preparation

1. Install necessary packages:
      ```shell
      pip install -r requirements.txt
      ```

2. Prepare the datasets. MeteoRA requires datasets in JSONL format. The tasks are primarily selected from the BIGBench dataset in the paper, which is in JSON format. To convert them to JSONL format, run:
      ```shell
      cd data
      python create_dataset.py --task all
      ```

   To create a specific dataset, use:
      ```shell
      cd data
      python create_dataset.py --task <task_name>
      ```

3. Prepare *composite-n* tasks. Refer to our paper for the definition of *composite-n* tasks. Generate these tasks using:
      ```shell
      python create_composite.py --n <n>
      ```
We prepared `n=3`, `n=5` and `n=10` few-shot dataset generating code. Before generation, please ensure that the sub-tasks to composite *composite-n* task have been included in `data/datasets`.

4. Prepare LoRA adapters and MeteoRA model checkpoints. You can train them yourself or download ours pre-trained models ([LlaMA2](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama2_13b) and [LlaMA3](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama3_8b) as base model):
      ```shell
      python download_ckpt.py
      ```

5. Update file paths in `configs/config.yaml`. Example paths:
      ```yaml
      base_model_path: 'meta-llama3/Meta-Llama-3-8B'
      meteora_ckpt_path: 'ckpt/llama3_8b/llama3_8b_meteora/top_2'
      adapter_dir: 'ckpt/llama3_8b/llama3_8b_peft'
      ```

### Evaluation

Run a benchmark with the MeteoRA model:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> 
```

For example:
```shell
python eval_model.py --task composite_10 --batch_size 4 
```

**Note:** For *composite-n* tasks, set a larger *temperature* value (`self.T` in `MoELoRA/layer.py`). Use `15`, `20`, and `30` for `n=3`, `n=5`, and `n=10`, respectively. For single tasks, use the default value (`self.T=1`).


To save the evaluation result:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --save
```

For debug mode (model output and ground truth will be shown in the console):
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --debug
```

Run a benchmark with the PEFT model:
```shell
python eval_model.py --task <task_name> --batch_size <batch_size> --model <adapter_name>
```

### Training the MeteoRA Model

0. Prepare LoRA adapters and corresponding datasets in JSONL format. Ensure each LoRA adapter has a corresponding dataset. Place all LoRA adapters and datasets in their respective folders with matching subfolder names:
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

1. Update file paths in `run_meteora_train_fsdp.sh`.


2. Train the MeteoRA model:
    ```shell
    sh run_meteora_train_fsdp.sh
    ```

**Note:** The current version of Triton acceleration supports inference mode only. Use the following settings when training the MeteoRA model:

```shell
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