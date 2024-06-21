import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tqdm import tqdm
import json
import argparse
import torch

import utils
from eval.eval import eval_benchmark
from eval.clean import clean_output
from eval.check import check_benchmark
from constant import MAX_SAMPLE

### file path
config = utils.load_config()
base_model_path = config['base_model_path']
adapter_dir = config['adapter_dir']
meteora_ckpt_path = config['meteora_ckpt_path']
eval_result_save_dir = config['eval_result_save_dir']
###

task_set = {
    "dir": "data/datasets",
    "tasks": {
        # 28 selected tasks
        "abstract_narrative_understanding", 
        "elementary_math_qa",
        "linguistics_puzzles",
        "strategyqa",
        "cnn_dailymail",
        "formal_fallacies_syllogisms_negation",
        "logical_deduction",
        "topical_chat",
        "contextual_parametric_knowledge_conflicts",
        "gsm8k",
        "object_counting",
        "vitaminc_fact_verification",
        "cs_algorithms",
        "language_identification",
        "question_selection",
        "alpaca",
        "news_commentary_de",
        "news_commentary_es",
        "news_commentary_it",
        "tracking_shuffled_objects",
        "goal_step_wikihow",
        "disfl_qa",
        "unit_conversion",
        "paragraph_segmentation",
        "reasoning_about_colored_objects",
        "epistemic_reasoning",
        "play_dialog_same_or_different",
        "winowhy",
        # composite tasks
        "composite_3",
        "composite_5",
        "composite_10",
    }
}

DEFAULT_MODEL = 'meteora'

def get_task_path(task_name):
    if task_name in task_set['tasks']:
        return os.path.join(task_set['dir'], task_name, 'test.jsonl')
    else:
        raise ValueError('Invalid task name')

def debug_inference(model, tokenizer, task_name):
    model_responses = []
    ground_truths = []
    cnt = 0

    gen_kwargs = dict(
        max_length=4096,
        no_repeat_ngram_size=30,
        pad_token_id=tokenizer.eos_token_id,
    )

    task_path = os.path.join(task_set['dir'], task_name, 'test.jsonl')
    with open(task_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            prompt = sample['prompt']
            input = tokenizer(prompt, return_tensors="pt", padding=True)
            # skip too long examples
            if input.input_ids.size(1) > 4096:
                continue
            input = input.to("cuda")
            cnt += 1

            print(f'\nSample {cnt}' + '-'*60 + '\n')
            output = model.generate(
                **input,
                **gen_kwargs,
            )
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            output = clean_output(output, prompt, task_name)
            model_responses.append(output)
            print(f'prompt: \n{prompt}')
            print(f'model output: \n{output}')

            ground_truth = sample['response']
            ground_truths.append(ground_truth)
            print('\nground truth: \n', ground_truth)

            if cnt == 10:
                break
    
    # stat
    result = {}
    result['eval_result'] = eval_benchmark(model_responses, ground_truths, task_name)
    result['warning'] = check_benchmark(model_responses, ground_truths, task_name)
    result['num_samples'] = len(model_responses)
    return result

def eval_single(model, tokenizer, task_name):

    model_responses = []
    ground_truths = []
    threshold = MAX_SAMPLE[task_name]

    gen_kwargs = dict(
        max_length=4096,
        no_repeat_ngram_size=30,
        pad_token_id=tokenizer.eos_token_id,
    )

    task_path = os.path.join(task_set['dir'], task_name, 'test.jsonl')
    with open(task_path, 'r') as f:
        lines = f.readlines()
        lines = lines[:threshold]
        pbar = tqdm(lines, total=len(lines))
        for line in pbar:
            sample = json.loads(line)
            prompt = sample['prompt']
            # skip too long examples
            if tokenizer(prompt, return_tensors="pt").input_ids.size(1) > 4096:
                continue
            input = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
            output = model.generate(
                **input,
                **gen_kwargs,
            )
            model_response = clean_output(tokenizer.decode(output[0], skip_special_tokens=True), 
                                          prompt ,task_name)
            
            model_responses.append(model_response)
            ground_truths.append(sample['response'])
    
    result = {}
    result['eval_result'] = eval_benchmark(model_responses, ground_truths, task_name)
    result['warning'] = check_benchmark(model_responses, ground_truths, task_name)
    result['num_samples'] = len(model_responses)
    return result

def eval_batch(model, tokenizer, task_name, batch_size):

    model_responses = []
    ground_truths = []
    cnt = 0
    batch = []

    max_sample = MAX_SAMPLE[task_name]

    gen_kwargs = dict(
        max_length=4096,
        no_repeat_ngram_size=30,
        pad_token_id=tokenizer.eos_token_id,
    )

    task_path = os.path.join(task_set['dir'], task_name, 'test.jsonl')
    with open(task_path, 'r') as f:
        lines = f.readlines()
        lines = lines[:max_sample]
        pbar = tqdm(lines, total=len(lines))
        for line in pbar:
            sample = json.loads(line)
            prompt = sample['prompt']
            # skip too long examples
            if tokenizer(prompt, return_tensors="pt").input_ids.size(1) > 4096:
                continue
            batch.append(prompt)
            ground_truths.append(sample['response']) 
            cnt += 1

            if cnt % batch_size == 0:
                with torch.no_grad():
                    inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
                    outputs = model.generate(
                        **inputs,
                        **gen_kwargs,
                    )
                for i, output in enumerate(outputs):
                    model_response = clean_output(tokenizer.decode(output, skip_special_tokens=True), 
                                                  batch[i] ,task_name)
                    model_responses.append(model_response)
                # reset batch
                batch = []
        
        # last batch
        if len(batch) > 0:
            with torch.no_grad():
                inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            for i, output in enumerate(outputs):
                model_response = clean_output(tokenizer.decode(output, skip_special_tokens=True), 
                                              batch[i] ,task_name)
                model_responses.append(model_response)

    result = {}
    result['eval_result'] = eval_benchmark(model_responses, ground_truths, task_name)
    result['warning'] = check_benchmark(model_responses, ground_truths, task_name)
    result['num_samples'] = len(model_responses)
    return result


if __name__ == '__main__':
    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=DEFAULT_MODEL, required=False)
    argparser.add_argument('--task', type=str, required=True)
    argparser.add_argument('--batch_size', type=int, default=1, required=False)
    argparser.add_argument('--save', required=False, action='store_true')
    argparser.add_argument('--debug', required=False, action='store_true')
    args = argparser.parse_args()

    result = {}
    result['args'] = vars(args)

    task_name = args.task
    tasks_path = get_task_path(task_name)

    # MeteoRA model
    if args.model == DEFAULT_MODEL:
        # TODO: remove this
        meteora_ckpt_path += '/model.safetensors'
        result['meteora_ckpt_path'] = meteora_ckpt_path
        model, tokenizer = utils.load_meteora_model(base_model_path, adapter_dir, meteora_ckpt_path)
    # PEFT
    else:
        adapter_path = os.path.join(adapter_dir, task_name)
        model, tokenizer = utils.load_peft_model(base_model_path, adapter_path)

    # debug mode
    if args.debug:
        debug_inference(model, tokenizer, task_name)
        exit(0)

    # eval
    batch_size = args.batch_size
    if batch_size == 1:
        result[task_name] = eval_single(model, tokenizer, task_name)
    else:
        result[task_name] = eval_batch(model, tokenizer, task_name, batch_size)

    # save
    if args.save:
        if args.model == DEFAULT_MODEL:
            save_path = os.path.join(eval_result_save_dir, 'meteora', f'{task_name}.json')
        else:
            save_path = os.path.join(eval_result_save_dir,'peft', f'{task_name}.json')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)