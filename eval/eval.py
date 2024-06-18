from functools import partial
from sacrebleu import BLEU
from rouge import Rouge
import re

def eval_rouge(model_responses, ground_truths):
    original_len = len(model_responses)
    try:
        # filter empty responses
        filtered_model_responses, filtered_ground_truths = zip(*[(x, y) for x, y in zip(model_responses, ground_truths) if len(x) > 0])
        model_responses = list(filtered_model_responses)
        ground_truths = list(filtered_ground_truths)
    except:
        # if all responses are empty
        print('ROUGE')
        print('0.00')
        print("Empty answer nums: ", original_len)
        return {"ROUGE": "Null"}
    
    new_len = len(model_responses)

    rouge = Rouge()
    rouge_results = rouge.get_scores(model_responses, ground_truths, avg=True)
    print('ROUGE')
    print(rouge_results)
    print("Empty answer nums: ", original_len - new_len)
    return {"ROUGE": rouge_results}

def eval_bleu(model_responses, ground_truths):
    original_len = len(model_responses)
    try:
        filtered_model_responses, filtered_ground_truths = zip(*[(x, y) for x, y in zip(model_responses, ground_truths) if len(x) > 0])
        model_responses = list(filtered_model_responses)
        ground_truths = list(filtered_ground_truths)
    except:
        # if all model responses are empty
        print('BLEU')
        print('0.00')
        print("Empty answer nums: ", original_len)
        return {"BLEU": "Null"}

    new_len = len(model_responses)

    predictions = model_responses
    ground_truths = [[ground_truth] for ground_truth in ground_truths]
    bleu = BLEU()
    bleu_results = bleu.corpus_score(predictions, ground_truths)
    print('BLEU')
    print(bleu_results)
    print("Empty answer nums: ", original_len - new_len)
    return {"BLEU": str(bleu_results)}

def eval_acc(model_responses, ground_truths, mode):
    correct = []
    for model_response, ground_truth in zip(model_responses, ground_truths):
        model_response = model_response.lower()
        ground_truth = ground_truth.lower()
        if mode == 'exact':
            correct.append(model_response == ground_truth)
        elif mode == 'fuzzy':
            correct.append(ground_truth in model_response)
        else:
            raise ValueError('Invalid accuracy mode')
        
    acc = sum(correct) / len(correct)
    print('Accuracy:', acc)
    return {"Accuracy": acc}

def eval_strategyqa_acc(model_responses, ground_truths):
    correct = []
    for model_response, ground_truth in zip(model_responses, ground_truths):
        model_response = model_response.lower()
        ground_truth = ground_truth.lower()
        if 'yes' in ground_truth[:5] and 'yes' in model_response[:5]:
            correct.append(True)
        elif 'no' in ground_truth[:5] and 'no' in model_response[:5]:
            correct.append(True)
        else:
            correct.append(False)

    acc = sum(correct) / len(correct)
    print('Accuracy:', acc)
    return {"Accuracy": acc}

def eval_gsm8k_acc(model_responses, ground_truths):
    INVALID_ANS = '[INVALID_ANS]'
    GSM8K_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def extract_answer_from_model_response(output):
        INVALID_ANS = '[INVALID_ANS]'
        ANSWER_TRIGGER = "The answer is"

        output = output.lower()
        preds = output.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        if pred[-1] == ".":
            pred = pred[:-1]

        return pred

    def extract_answer_from_ground_truth(ground_truth):
        match = GSM8K_ANS_RE.search(ground_truth)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "") # remove ',' in numbers
            return match_str
        else:
            return INVALID_ANS

    correct = []
    for model_response, ground_truth in zip(model_responses, ground_truths):
        model_response = extract_answer_from_model_response(model_response)
        ground_truth = extract_answer_from_ground_truth(ground_truth)
        assert ground_truth != INVALID_ANS
        correct.append(model_response == ground_truth)

    acc = sum(correct) / len(correct)
    print('Accuracy:', acc)
    return {"Accuracy": acc}

def eval_composite(model_responses, ground_truths, tasks):
    result = {}
    model_responses_splits = []
    empty_cnt = 0

    # exract model responses for each task
    for model_response in model_responses:
        model_response_splits = []
        # all tasks
        for i in range(len(tasks)):
            task_flag = str(i + 1) + '. '
            if task_flag not in model_response:
                empty_cnt += 1
                model_response_splits.append('')
            else:
                start = task_flag
                end = r'(?:\n|$)'
                pattern = re.escape(start) + r'(.*?)' + end
                match = re.search(pattern, model_response)
                if match:
                    extracted_content = match.group(1)
                    if len(extracted_content) == 0:
                        empty_cnt += 1
                    model_response_splits.append(extracted_content)
                else:
                    model_response_splits.append('')
        model_responses_splits.append(model_response_splits)

    avg_attempt_nums = len(tasks) - empty_cnt / len(model_responses)
    print("Avg attempt nums: ", avg_attempt_nums, '\n')
    result['Avg attempt nums'] = avg_attempt_nums

    avg_correct_nums = 0
    ground_truths_splits = [ground_truth.split('@@@@\n') for ground_truth in ground_truths]
    for i, task in enumerate(tasks):
        print('############')
        print(task)
        result[task] = []
        for fn in EVAL_FN[task]:
            model_responses_current = [model_response[i] for model_response in model_responses_splits]
            ground_truths_current = [ground_truth[i] for ground_truth in ground_truths_splits]
            current_result = fn(model_responses_current, ground_truths_current)
            avg_correct_nums += current_result.get('Accuracy', 0)
            result[task].append(current_result)
            print()

    print('Avg correct nums: ', avg_correct_nums)
    result['Avg correct nums'] = avg_correct_nums

    return result

EVAL_FN = {
    "formal_fallacies_syllogisms_negation": [partial(eval_acc, mode='exact')],
    "language_identification": [partial(eval_acc, mode='fuzzy')],
    "linguistics_puzzles": [eval_bleu, eval_rouge],
    "logical_deduction": [partial(eval_acc, mode='fuzzy')],
    "play_dialog_same_or_different": [partial(eval_acc, mode='fuzzy')],
    "strategyqa": [eval_bleu, eval_rouge, eval_strategyqa_acc],
    "vitaminc_fact_verification": [partial(eval_acc, mode='fuzzy')],
    "winowhy": [partial(eval_acc, mode='exact')],
    "abstract_narrative_understanding": [partial(eval_acc, mode='fuzzy')],
    "elementary_math_qa": [partial(eval_acc, mode='exact')],
    "cnn_dailymail": [eval_bleu, eval_rouge],
    "topical_chat": [eval_bleu, eval_rouge],
    "contextual_parametric_knowledge_conflicts": [partial(eval_acc, mode='fuzzy')],
    "gsm8k": [eval_gsm8k_acc],
    "object_counting": [partial(eval_acc, mode='fuzzy')],
    "cs_algorithms": [partial(eval_acc, mode='exact')],
    "question_selection": [partial(eval_acc, mode='fuzzy')],
    "alpaca": [eval_bleu, eval_rouge],
    "news_commentary_es": [eval_bleu],
    "news_commentary_it": [eval_bleu],
    "news_commentary_de": [eval_bleu],
    "tracking_shuffled_objects": [partial(eval_acc, mode='exact')],
    "goal_step_wikihow": [partial(eval_acc, mode='fuzzy')],
    "disfl_qa": [partial(eval_acc, mode='fuzzy')],
    "unit_conversion": [partial(eval_acc, mode='exact')],
    "paragraph_segmentation": [partial(eval_acc, mode='exact')],
    "reasoning_about_colored_objects": [partial(eval_acc, mode='fuzzy')],
    "epistemic_reasoning": [partial(eval_acc, mode='exact')],

    # composite tasks
    'composite_3': [partial(eval_composite, 
                            tasks=["logical_deduction", "question_selection", "strategyqa"])],
    'composite_5': [partial(eval_composite, 
                            tasks=["logical_deduction", "question_selection", "abstract_narrative_understanding", "goal_step_wikihow", "strategyqa"])],
    'composite_10': [partial(eval_composite, 
                             tasks=["logical_deduction", "question_selection", "abstract_narrative_understanding", "goal_step_wikihow", "winowhy", "strategyqa", "disfl_qa", "news_commentary_de", "alpaca", "linguistics_puzzles"])],
}

def eval_benchmark(model_responses, ground_truths, task_name):
    print("\n\n############### Evaluation Result ###############")
    print("Task: ", task_name)

    result = []
    for fn in EVAL_FN[task_name]:
        print()
        result.append(fn(model_responses, ground_truths))
    return result