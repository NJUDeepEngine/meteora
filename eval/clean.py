import re

def get_prompt_end(prompt):
    end_pattern = '[/INST]\n'
    return end_pattern

def clean_default(output, prompt):
    if prompt not in output:
        prompt_end = get_prompt_end(prompt)
        idx = output.rfind(prompt_end)
        if idx == -1:
            print('--------------- Warning: Fixing prompt format failed. ---------------')
            return output
        output = output[idx+len(prompt_end):]
        output = output.lower()
        return output
    else:
        output = output.strip()
        prompt = prompt.strip()
        output = output.replace(prompt, '').strip()
        output = output.lower()
        return output

def clean_multi_choice(output, prompt):
    output = clean_default(output, prompt)
    return output

def clean_disfl_qa(output, prompt):
    # find the last '\n' in the output
    idx = output.rfind('\n')
    if idx == -1:
        return 'illegal answer'
    output = output[idx+1:]
    output = output.lower()
    return output

def clean_formal_fallacies_syllogisms_negation(output, prompt):
    output = clean_default(output, prompt)
    # only 'valid' and 'invalid' are legal answers
    if (len(re.findall('invalid', output)) >= 1):
        output = 'invalid'
    elif (len(re.findall('valid', output)) >= 1):
        output = 'valid'
    else:
        output = 'illegal answer'
    return output

def clean_vitaminc_fact_verification(output, prompt):
    output = clean_default(output, prompt)
    # only 'true', 'false' and 'neither' are legal answers
    if (len(re.findall('true', output)) >= 1):
        output = 'true'
    elif (len(re.findall('false', output)) >= 1):
        output = 'false'
    elif (len(re.findall('neither', output)) >= 1):
        output = 'neither'
    else:
        output = 'illegal answer'
    return output

def clean_gsm8k(output, prompt):
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

CLEAN_FN = {
    "formal_fallacies_syllogisms_negation": clean_formal_fallacies_syllogisms_negation,
    "language_identification": clean_multi_choice,
    "linguistics_puzzles": clean_default,
    "logical_deduction": clean_multi_choice,
    "play_dialog_same_or_different": clean_multi_choice,
    "strategyqa": clean_default,
    "vitaminc_fact_verification": clean_multi_choice,
    "winowhy": clean_multi_choice,
    "abstract_narrative_understanding": clean_multi_choice,
    "elementary_math_qa": clean_multi_choice,
    "cnn_dailymail": clean_default,
    "topical_chat": clean_default,
    "contextual_parametric_knowledge_conflicts": clean_multi_choice,
    "gsm8k": clean_default,
    "object_counting": clean_multi_choice,
    "cs_algorithms": clean_multi_choice,
    "question_selection": clean_multi_choice,
    "alpaca": clean_default,
    "news_commentary_es": clean_default,
    "news_commentary_it": clean_default,
    "news_commentary_de": clean_default,
    "tracking_shuffled_objects": clean_multi_choice,
    "goal_step_wikihow": clean_multi_choice,
    "disfl_qa": clean_default, 
    "unit_conversion": clean_multi_choice,
    "paragraph_segmentation": clean_default,
    "reasoning_about_colored_objects": clean_default,
    "epistemic_reasoning": clean_multi_choice,
    "composite_3": clean_default,
    "composite_5": clean_default,
    "composite_10": clean_default,
}

def clean_output(output, prompt, task):
    return CLEAN_FN[task](output, prompt)