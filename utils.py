import random
import torch
import os
import datasets
from typing import List, Dict, Any
from functools import partial
import copy
import yaml

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    LlamaForCausalLM,
)
from peft import PeftModel as HF_PeftModel

from base_model.llama.modeling_llama_meteor import LlamaMeteorForCausalLM, LlamaMeteorModel
import torch
from MoELoRA.peft_model import PeftModel
import torch.distributed._shard.checkpoint as dist_cp

from constant import METEORA_TASKS

IGNORE_INDEX = -100

def tokenize_dataset(input: Dict[str, str], tokenizer, max_length=None, task_index=None) -> Dict[str, Any]:
    # This special index is used to ignore certain tokens during loss calculation.
    input_ids, attention_mask, labels, gate_labels = [], [], [], []
    keys = ["prompt", "response"]
    for key in keys:
        text = input[key]
        msg_tokenized = tokenizer(
            text,
            truncation=False, # truncate later
            add_special_tokens=False, # since we already added the chatml style special tokens manually
        )
        input_ids += msg_tokenized['input_ids']
        attention_mask += msg_tokenized['attention_mask']
        # labels += [IGNORE_INDEX] * len(msg_tokenized['input_ids']) if key == "prompt" else msg_tokenized['input_ids']
        # gate_labels += [IGNORE_INDEX] * len(msg_tokenized['input_ids']) if key == "prompt" else [data_index] * len(msg_tokenized['input_ids'])        
        labels += msg_tokenized['input_ids'] if key == "prompt" else msg_tokenized['input_ids']
        gate_labels += [task_index] * len(msg_tokenized['input_ids']) if key == "prompt" else [task_index] * len(msg_tokenized['input_ids'])        
    final_labels = [labels, gate_labels]
    # truncate here
    final_labels[0] = final_labels[0][:max_length]
    final_labels[1] = final_labels[1][:max_length]
    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": final_labels
    }

def collate_dataset(samples: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    """collate the dataset to a batch

    Args:
        samples (List[Dict[str, Any]]): [{
            "input_ids": ...,
            "attention_mask": ...,
            "labels": ...
        }]
    """
    max_len = max([len(s['input_ids']) for s in samples])
    max_len_clip = max_len
    if max_len > 4096:
        warnings.warn(f"The max length of a single sample is {max_len}, which exceeds the maximum sequence length of 4096.")
        max_len_clip = 4096

    # print([len(s['input_ids']) for s in samples])
    # print(max_len)
    batch_samples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    pad_elems = {
        "input_ids": tokenizer.pad_token_id, # append with <PAD> token to align
        "attention_mask": 0, # append with 0 to ignore the padding tokens
        "labels": IGNORE_INDEX # append with -100 to ignore them during loss calculation
    }
    for sample in samples:
        # if max_len > max_len_clip:
        #     sample = sample[:]
        # print(max_len,len(sample['input_ids']))
        pad_len = max_len - len(sample['input_ids'])
        # padding each sample to align with the longest one
        for k in sample:
            if k == "labels":
                batch_samples[k].append([sample[k][0] + [pad_elems[k]] * pad_len, sample[k][1] + [pad_elems[k]] * pad_len])
            else:
                batch_samples[k].append(
                    sample[k] + [pad_elems[k]] * pad_len
                )
            # print(len(batch_samples['input_ids'][k]), len(batch_samples['attention_mask'][k]), len(batch_samples['labels'][k][0]), len(batch_samples['labels'][k][1]))
    # the dtype should be torch.long or torch.int64, but it is not necessary since the default dtype for a int list is just int64
    # print(len(batch_samples['input_ids'][0]), len(batch_samples['input_ids'][1]), len(batch_samples['labels'][0][1]), len(batch_samples['labels'][1][1]))
    # print(max_len)
    batch = {k: torch.tensor(v, dtype=torch.long)  for k,v in batch_samples.items()} 
    # print(batch["input_ids"].size(), batch["labels"].size())
    # print(batch["attention_mask"])
    return batch

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(
                self.trainer.deepspeed
            )
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def tokenize_datasets(dataset, tokenizer, max_length, dataset_type, data_index):
    task_id = copy.copy(data_index)
    print(dataset.map, task_id)
    
    dataset_tokenized = dataset.map(
        partial(tokenize_dataset, tokenizer=tokenizer, max_length=max_length, task_index=task_id),
        batched=False,
        num_proc=4,
        # num_proc=os.cpu_count(),    # multi-threaded with all cpu cores
        # remove_columns=dataset["dataset_type"].column_names  # don't need this anymore, we have tokens from here on
    )
    print(dataset_tokenized)
    return dataset_tokenized


def get_dataset_name_from_tasks_path(path):
    dataset_names = [name for name in os.listdir(path) if name != "alpaca"]
    return dataset_names


def create_bbl_united_dataset(tasks, tokenizer, max_length, default_task):

    trains = []
    tests = []
    data_index = 0
    for task in tasks:
        
        if task == default_task:
            dataset = datasets.load_dataset('json', data_files=task+"/train.jsonl",split='train')
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.train_test_split(test_size=0.5)
            train_dataset = tokenize_datasets(dataset['train'], tokenizer, max_length, "train", data_index)
            test_dataset = tokenize_datasets(dataset['test'], tokenizer, max_length, "test", data_index)
            # train_dataset = dataset["train"].select(range(10000))
            # train_dataset = tokenize_datasets(train_dataset, tokenizer, max_length, "train", data_index)
            # test_dataset = dataset["test"].select(range(2000))
            # test_dataset = tokenize_datasets(test_dataset, tokenizer, max_length, "test", data_index)
            
        else:
            data_files = {"train": task+"/train.jsonl", "test": task+"/test.jsonl"}
            dataset = datasets.load_dataset('json', data_files=data_files)
        
            train_dataset = tokenize_datasets(dataset['train'], tokenizer, max_length, "train", data_index)
            test_dataset = tokenize_datasets(dataset['test'], tokenizer, max_length, "test", data_index)
        trains.append(train_dataset)
        tests.append(test_dataset)
        data_index += 1
    
    
    merged_train_dataset = datasets.concatenate_datasets(trains)
    shuffled_train_dataset = merged_train_dataset.shuffle(seed=42)
    merged_dataset = shuffled_train_dataset.train_test_split(test_size=0.1)
    print("dataset is loaded, train:", merged_dataset["train"], "\ntest:", merged_dataset["test"], merged_dataset)
    return merged_dataset["train"], merged_dataset["test"]
    # merged_test_dataset = datasets.concatenate_datasets(tests)
    # merged_train_dataset = datasets.concatenate_datasets([gsm8k_dataset['train'], sqlctx_dataset['train'], viggo_dataset['train']])
    # merged_test_dataset = datasets.concatenate_datasets([gsm8k_dataset['test'], sqlctx_dataset['test'], viggo_dataset['test']])

    # print("dataset is loaded, train:", shuffled_train_dataset, "test:", merged_test_dataset)
    # merged_train_dataset = tokenize_datasets(merged_train_dataset, tokenizer, max_length, "train")
    # merged_test_dataset = tokenize_datasets(merged_test_dataset, tokenizer, max_length, "test")
    
    # return shuffled_train_dataset, merged_test_dataset


def load_model_and_tokenizer(model_path, tasks_datasets_prefix, lora_path_prefix, default_task="alpaca"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = LlamaMeteorForCausalLM.from_pretrained(model_path,                                                           
                                                          device_map=None,
                                                          torch_dtype=torch.bfloat16, 
                                                          attn_implementation="flash_attention_2"
                                                          )
    
    # tasks_datasets_prefix = task_path
    tasks = get_dataset_name_from_tasks_path(tasks_datasets_prefix)
    default_task = "alpaca"
    # tasks.append(default_task)
    ADAPTERS = { "lora" + str(index+1):
                lora_path_prefix + task + "_no_sys" for index, task in enumerate(tasks) }

    model = PeftModel.from_pretrained_multi(
        model=base_model, 
        adapter_names=ADAPTERS, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        is_trainable=False,
    )

    return model, tokenizer, tasks


def load_meteora_model(weight_path, model_path, tasks_datasets_prefix, lora_path_prefix):
    # get model(without weights) and tokenizer
    model, tokenizer, tasks = load_model_and_tokenizer(model_path, tasks_datasets_prefix, lora_path_prefix)
    tokenizer.pad_token = tokenizer.eos_token

    # overwrite the model with sft result
    print('weight path: ', weight_path)
    state_dict = {
        "model": model.state_dict()
    }
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(weight_path),
        no_dist=True
    )
    model.load_state_dict(state_dict['model'])

    return model, tokenizer, tasks


def create_gsm8k_vggio_sqlctx(data_path_prefix, tokenizer, max_length):

    gsm8k_data_files = {"train": data_path_prefix+"gsm8k-train.jsonl", "test": data_path_prefix+"gsm8k-test.jsonl"}
    viggo_data_files = {"train": data_path_prefix+"viggo-train.jsonl", "test": data_path_prefix+"viggo-test.jsonl"}

    gsm8k_dataset = datasets.load_dataset('json', data_files=gsm8k_data_files)
    
    data_index = 0
    
    gsm8k_train_dataset = tokenize_datasets(gsm8k_dataset['train'], tokenizer, max_length, "train", data_index)
    gsm8k_test_dataset = tokenize_datasets(gsm8k_dataset['test'], tokenizer, max_length, "test", data_index)
    
    data_index += 1
    
    viggo_dataset = datasets.load_dataset('json', data_files=viggo_data_files)
    viggo_train_dataset = tokenize_datasets(viggo_dataset['train'], tokenizer, max_length, "train", data_index)
    viggo_test_dataset = tokenize_datasets(viggo_dataset['test'], tokenizer, max_length, "test", data_index)
    
    data_index += 1
    
    sqlctx_raw_dataset = datasets.load_dataset('json', data_files=data_path_prefix+"sqlctx-train.jsonl")
    sqlctx_dataset = sqlctx_raw_dataset['train'].train_test_split(test_size=0.1)

    sqlctx_train_dataset = tokenize_datasets(sqlctx_dataset['train'], tokenizer, max_length, "train", data_index)
    sqlctx_test_dataset = tokenize_datasets(sqlctx_dataset['test'], tokenizer, max_length, "test", data_index)
    
    merged_train_dataset = datasets.concatenate_datasets([gsm8k_train_dataset, sqlctx_train_dataset, viggo_train_dataset])
    merged_test_dataset = datasets.concatenate_datasets([gsm8k_test_dataset, sqlctx_test_dataset, viggo_test_dataset]) 
    # merged_train_dataset = datasets.concatenate_datasets([gsm8k_dataset['train'], sqlctx_dataset['train'], viggo_dataset['train']])
    # merged_test_dataset = datasets.concatenate_datasets([gsm8k_dataset['test'], sqlctx_dataset['test'], viggo_dataset['test']])

    print("dataset is loaded, train:", merged_train_dataset, "test:", merged_test_dataset)
    # merged_train_dataset = tokenize_datasets(merged_train_dataset, tokenizer, max_length, "train")
    # merged_test_dataset = tokenize_datasets(merged_test_dataset, tokenizer, max_length, "test")
    
    return merged_train_dataset, merged_test_dataset

def create_datasets(tokenizer, dataset_name, args):
    dataset = load_dataset(
        dataset_name, token=True, num_proc=args.num_workers
    )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=True,
        add_eos_token=False,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=False,
        add_eos_token=False,
    )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        # use_flash_attention_2=args.use_flash_attn
        attn_implementation="sdpa" if args.use_flash_attn else "eager",
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if (
            args.use_4bit_qunatization or args.use_8bit_qunatization
        ) and args.use_peft_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.use_gradient_checkpointing
            )

        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model, peft_config


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

def load_peft_model(base_model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = LlamaForCausalLM.from_pretrained(base_model_path,
                                                    device_map="auto",
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2"
                                                    )
    
    peft_model = HF_PeftModel.from_pretrained(base_model, 
                                        model_id=adapter_path,
                                        adapter_name=adapter_path, 
                                        torch_dtype=torch.bfloat16, 
                                        attn_implementation="flash_attention_2", 
                                        is_trainable=False)
    
    return peft_model, tokenizer

def load_meteora_model(base_model_path, adapter_dir, meteora_ckpt_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = LlamaMeteorForCausalLM.from_pretrained(base_model_path, 
                                                          device_map='auto', 
                                                          torch_dtype=torch.bfloat16, 
                                                          attn_implementation="flash_attention_2"
                                                          )
    tasks = METEORA_TASKS
    adapters = { task: 
                 os.path.join(adapter_dir, task) for index, task in enumerate(tasks)}
    meteora_model = PeftModel.from_pretrained_multi(
        model=base_model, 
        adapter_names=adapters, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        is_trainable=False,
    )

    # load ckpt
    state_dict = {
        "model": meteora_model.state_dict()
    }
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(meteora_ckpt_path),
        no_dist=True
    )
    meteora_model.load_state_dict(state_dict['model'])

    return meteora_model, tokenizer

def load_config():
    with open(os.path.join('configs', 'config.yaml')) as f:
        try:
            config = yaml.safe_load(f)
        except:
            print("Error loading config file")
            return None
    return config