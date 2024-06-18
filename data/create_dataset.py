import argparse
import os
import json
import datasets
import random
from tqdm import tqdm

from dataset_presets import get_presets
from dataset_utils import process_sample, get_prompt_and_response_hf

### file path
bigbench_dataset_dir = ""
target_dir = "datasets"
###

random.seed(42)

# prevent too large dataset
SAMPLE_NUMS_THRESHOLD = {
    'train': 100000,
    'validation': 50000,
    'test': 50000,
}

def devide_train_split(train_split_path, test_split_path, test_size=0.2):
    with open(train_split_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # shuffle
    random.shuffle(lines)
    # devide
    nums_test = int(len(lines) * test_size)
    train_split = lines[nums_test:]
    test_split = lines[:nums_test]
    # write files
    with open(train_split_path, 'w', encoding='utf-8') as f:
        f.writelines(train_split)
    with open(test_split_path, 'w', encoding='utf-8') as f:
        f.writelines(test_split)

def cerate_benchmark_hf(benchmark, presets):
    splits = ['train', 'test']
    preset = presets[benchmark]
    prompt_template = preset['prompt']
    response_template = preset['response']
    dataset_name = preset['dataset_name']
    subset_name = preset['subset_name']

    for split in splits:
        target_file = os.path.join(target_dir, benchmark, f"{split}.jsonl")
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        target = open(target_file, 'w')
        cnt = 0

        # check whether original split exists
        dataset = datasets.load_dataset(dataset_name, subset_name)
        if split not in dataset.keys():
            print(f"Warning: Split {split} not found in dataset {dataset_name}/{subset_name}, divide 20% of train split as test split.")
            train_split_path = target_file.replace('test', 'train')
            test_split_path = target_file
            devide_train_split(train_split_path, test_split_path)
            return

        raw_dataset = dataset[split]

        for sample in raw_dataset:
            cnt += 1
            prompt, response = get_prompt_and_response_hf(benchmark, sample, prompt_template, response_template)
            # jsonl format
            target.write(json.dumps({"prompt": prompt, "response": response}))
            target.write("\n")
            if cnt == SAMPLE_NUMS_THRESHOLD[split]:
                break

        target.close()


def cerate_benchmark_bigbench_json(benchmark, presets):
    # There is not a test split in BigBench dataset, so we use validation split as test split.
    splits = ['train', 'validation']
    prompt_template = presets[benchmark]['prompt']
    response_template = presets[benchmark]['response']
    rename = {
        'train': 'train',
        'validation': 'test',
    }

    for split in splits:
        cnt = 0
        source = open(os.path.join(bigbench_dataset_dir, benchmark, f"{split}.jsonl"), 'r')
        target_file = os.path.join(target_dir, benchmark, f"{rename[split]}.jsonl")
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        target = open(target_file, 'w')

        for line in source:
            cnt += 1
            sample = json.loads(line)
            sample['targets'] = sample['targets'][0]
            sample = process_sample(benchmark, sample)
            prompt = prompt_template.format(**sample)
            response = response_template.format(**sample)
            # jsonl format
            target.write(json.dumps({"prompt": prompt, "response": response}))
            target.write("\n")
            if cnt == SAMPLE_NUMS_THRESHOLD[split]:
                break

        source.close()
        target.close()

def cerate_benchmark(benchmark, presets):
    format = presets[benchmark]['format']
    if format == 'bigbench':
        cerate_benchmark_bigbench_json(benchmark, presets)
    if format == 'hf':
        cerate_benchmark_hf(benchmark, presets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    presets = get_presets()

    if args.task == 'all':
        for benchmark in tqdm(presets.keys()):
            cerate_benchmark(benchmark, presets)
    else:
        cerate_benchmark(args.task, presets)