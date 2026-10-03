"""Calibrate predictor thresholds on independently reviewed val event onsets."""
from report import key, read_jsonl, read_labels

SIGNALS = ('u_r_norm', 'u_f_norm', 'u_s')


def onset_examples(steps_path, labels_path, partition, tasks):
    labels = read_labels(labels_path)
    examples = []
    for step in read_jsonl(steps_path):
        if step['partition'] != partition or step['task'] not in tasks:
            continue
        label = labels.get(key(step))
        if label is None:
            continue
        onset = label['onset']
        transition = step['transition']
        if onset is not None and transition > onset:
            continue
        for signal in SIGNALS:
            if signal not in step:
                raise ValueError(f'missing {signal}; rerun rollouts with predictor logging')
        examples.append((step, transition == onset))
    if not examples:
        raise ValueError('no reviewed onset examples')
    return examples


def counts(examples, signal, threshold):
    tp = fp = fn = tn = 0
    for step, actual in examples:
        predicted = step[signal] >= threshold
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def calibrate(examples):
    if not any(actual for _, actual in examples) or not any(not actual for _, actual in examples):
        raise ValueError('need positive and negative reviewed val onset examples')
    result = {}
    for signal in SIGNALS:
        values = sorted(set(float(step[signal]) for step, _ in examples))
        candidates = [values[0]] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1e-9]
        threshold = max(candidates, key=lambda t: (counts(examples, signal, t)['f1'], t))
        result[signal] = dict(threshold=threshold,
                              validation=counts(examples, signal, threshold))
    return result


def evaluate(examples, calibration):
    return {signal: counts(examples, signal, calibration[signal]['threshold'])
            for signal in SIGNALS}
