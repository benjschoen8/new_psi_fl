"""Independent ground-truth relation and global-classifier evaluation."""
from collections import defaultdict
from itertools import combinations
from math import sqrt


def semantic_alignment(predicted, truth):
    """Map each pure predicted group to its truth class; ambiguous merges abstain.

    This does not fit a permutation to test predictions. It uses label metadata
    only. A predicted class merging distinct true classes is always incorrect.
    """
    classes = defaultdict(set)
    for key, labels in predicted.items():
        for label, gid in labels.items():
            classes[gid].add(truth[key][label])
    return {gid: next(iter(values)) for gid, values in classes.items() if len(values) == 1}


def mapping_metrics(predicted, truth):
    """Cross-group label-pair metrics; retain the legacy balanced accuracy/MCC.

    With identical truth, predicted relations and pair scope, these match
    evaluate_mapping_results. Zero denominators return zero as in the baseline.
    pair_accuracy is ordinary accuracy, distinct from legacy AvgAccuracy.
    """
    tp = fp = tn = fn = 0
    for a, b in combinations(truth, 2):
        for la, ta in truth[a].items():
            for lb, tb in truth[b].items():
                pa, pb = predicted.get(a, {}).get(la), predicted.get(b, {}).get(lb)
                match = pa is not None and pb is not None and pa == pb
                if ta == tb:
                    tp += int(match)
                    fn += int(not match)
                else:
                    fp += int(match)
                    tn += int(not match)
    div = lambda a, b: a / b if b else 0.
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    mcc_denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                precision=precision, recall=recall, f1=div(2 * precision * recall, precision + recall),
                specificity=specificity, balanced_accuracy=(recall + specificity) / 2,
                mcc=div(tp * tn - fp * fn, mcc_denominator),
                pair_accuracy=div(tp + tn, tp + fp + tn + fn), pairs=tp + fp + tn + fn)


def old_acc(predicted_ids, local_labels, local_to_global):
    """Original global accuracy: predicted mapping defines the test targets.

    Unmapped labels are excluded from the denominator, as in the old server.
    Returns an unrounded fraction; multiply by 100 to compare old CSV values.
    """
    if len(predicted_ids) != len(local_labels):
        raise ValueError('Predictions and labels must have equal lengths')
    correct = samples = 0
    for prediction, label in zip(predicted_ids, local_labels):
        label = int(label)
        if label in local_to_global:
            samples += 1
            correct += int(int(prediction) == local_to_global[label])
    return dict(accuracy=correct / samples if samples else 0., correct=correct, samples=samples)


def evaluate_global(model, datasets, predicted_mapping, ground_truth, device='cpu'):
    """datasets: iterable of (dataset name, group key, labeled test loader).

    Both metrics use the same forward-pass predictions. Ground-truth accuracy
    counts all test samples; old_acc excludes unmapped labels/groups. Unknown
    ground-truth labels are errors. All accuracies are fractions, not percentages.
    The existing accuracy field remains an alias of ground_truth_acc.
    """
    import torch
    alignment = semantic_alignment(predicted_mapping, ground_truth)
    was_training = model.training
    model.to(device).eval()
    scores = defaultdict(lambda: {'correct': 0, 'samples': 0, 'old_correct': 0, 'old_samples': 0})
    ambiguous = 0
    try:
        with torch.no_grad():
            for name, group, loader in datasets:
                for x, labels in loader:
                    output = model(x.to(device))
                    logits = output[1] if isinstance(output, tuple) else output
                    predictions = logits.argmax(dim=1).cpu().tolist()
                    local_labels = labels.tolist()
                    legacy = old_acc(predictions, local_labels, predicted_mapping.get(group, {}))
                    scores[name]['old_correct'] += legacy['correct']
                    scores[name]['old_samples'] += legacy['samples']
                    for gid, label in zip(predictions, local_labels):
                        target = ground_truth[group][int(label)]
                        scores[name]['samples'] += 1
                        scores[name]['correct'] += int(alignment.get(gid) == target)
                        ambiguous += int(gid not in alignment)
    finally:
        model.train(was_training)
    total = sum(row['samples'] for row in scores.values())
    if not total:
        raise ValueError('Evaluation requires nonempty test data')
    correct = sum(row['correct'] for row in scores.values())
    old_correct = sum(row['old_correct'] for row in scores.values())
    old_samples = sum(row['old_samples'] for row in scores.values())
    for row in scores.values():
        row['accuracy'] = row['correct'] / row['samples'] if row['samples'] else 0.
        row['ground_truth_acc'] = row['accuracy']
        row['old_acc'] = row['old_correct'] / row['old_samples'] if row['old_samples'] else 0.
    return dict(accuracy=correct / total, ground_truth_acc=correct / total,
                old_acc=old_correct / old_samples if old_samples else 0.,
                correct=correct, samples=total, old_correct=old_correct, old_samples=old_samples,
                ambiguous_predictions=ambiguous, by_dataset=dict(scores),
                scope='supplied_test_partitions',
                accuracy_unit='fraction', old_alignment_policy='predicted_mapping_targets',
                alignment_policy='pure_semantic_class_only')
