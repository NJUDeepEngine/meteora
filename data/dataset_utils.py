from dataset_presets import news_commentary

def process_sample(benchmark, sample):
    '''
    Process sample for some special benchmark.
    '''

    if benchmark == 'logical_deduction':
        # append choices
        sample['inputs'] += ' Which choice is correct?\n'
        for choice in sample['multiple_choice_targets']:
            sample['inputs'] += f'choice: {choice}\n'

    if benchmark == 'abstract_narrative_understanding':            
        # append choices
        sample['inputs'] += '\n'
        for choice in sample['multiple_choice_targets']:
            sample['inputs'] += f'choice: {choice}\n'

    if benchmark == 'tracking_shuffled_objects':
        # append choices
        sample['inputs'] += '\n'
        for choice in sample['multiple_choice_targets']:
            sample['inputs'] += f'choice: {choice}\n'

    if benchmark == 'unit_conversion':
        # append choices
        sample['inputs'] += '\n'
        for choice in sample['multiple_choice_targets']:
            sample['inputs'] += f'choice: {choice}\n'

    return sample

def get_prompt_and_response_hf(benchmark, sample, prompt_template, response_template):
    # news_commentary
    if benchmark in news_commentary:
        sample = sample['translation']
        prompt = prompt_template.format(**sample)
        response = response_template.format(**sample)
    # alpaca
    if benchmark == 'alpaca':
        # appned input if exists
        if len(sample['input']) > 0:
            sample['instruction'] += '\n' + sample['input']
        prompt = prompt_template.format(**sample)
        response = response_template.format(**sample)
    # default
    else:
        prompt = prompt_template.format(**sample)
        response = response_template.format(**sample)
    return prompt, response