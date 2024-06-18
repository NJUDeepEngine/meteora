def check_zero_len(model_responses, ground_truths):
    cnt = 0
    for i in range(len(model_responses)):
        if len(model_responses[i]) == 0:
            cnt += 1
    if cnt > 0:
        print('Zero length response count: ', cnt)
    return {"zero length response count: ": cnt}

def check_too_long(model_responses, ground_truths):
    cnt = 0
    for i in range(len(model_responses)):
        response_words = model_responses[i].split()
        ground_truth_words = ground_truths[i].split()
        if len(response_words) / len(ground_truth_words) > 2:
            cnt += 1
    if cnt > 0:
        print('Too long response count: ', cnt)
    return {"too long response count: ": cnt}


def check_benchmark(model_responses, ground_truths, task):
    warnings = []
    check_fn = [check_zero_len, check_too_long]
    for fn in check_fn:
        warnings.append(fn(model_responses, ground_truths))
    return warnings