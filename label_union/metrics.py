"""How good is a label union list? Scored over the public dictionary, one decision per entry:
"is this label in the union?"  (not label pairs: that is label mapping, evaluation.mapping_metrics)

  TP = in predicted and true union     FP = predicted only
  FN = true only (a label lost)        TN = in neither
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
When a factor is 0 (e.g. the dictionary is exactly the union, so TN = 0) MCC is undefined:
mcc_defined is False and mcc falls back to 1.0 / 0.0 for exact / not exact. Then MCC is only an
exactness flag: report exact, split/merged/unused indices and recall instead.
holders_exact: the per-label holder counts the Aggregator learned also match the truth.
"""
from math import sqrt


def union_metrics(predicted, truth, dictionary, predicted_holders=None, true_holders=None):
    """predicted, truth: iterables of label ids; dictionary: the public list they come from.
    predicted_holders / true_holders: optional {label: number of clients holding it}."""
    predicted, truth, dictionary = set(predicted), set(truth), set(dictionary)
    if not predicted | truth <= dictionary:
        raise ValueError(f'labels outside the dictionary: {sorted((predicted | truth) - dictionary)}')
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    tn = len(dictionary) - tp - fp - fn
    div = lambda a, b: a / b if b else 0.
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    exact = fp == fn == 0
    out = dict(true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
               precision=precision, recall=recall, f1=div(2 * precision * recall, precision + recall),
               mcc=(tp * tn - fp * fn) / den if den else float(exact), mcc_defined=bool(den),
               jaccard=div(tp, tp + fp + fn),
               exact=exact, missing=sorted(truth - predicted), spurious=sorted(predicted - truth),
               union_size=len(predicted), true_union_size=len(truth), dictionary_size=len(dictionary))
    if predicted_holders is not None and true_holders is not None:
        out['holders_exact'] = dict(predicted_holders) == dict(true_holders)
    return out


def true_union(client_labels):
    """Ground truth (experimenter only): every label some client holds, and how many hold it."""
    holders = {}
    for labels in client_labels:
        for x in set(labels):
            holders[x] = holders.get(x, 0) + 1
    return set(holders), holders


def index_metrics(client_labels, client_index, U, dictionary):
    """Score the index outputs of label_union.dict_union (experimenter only).

    client_labels: each client's label ids; client_index: each client's {label: index};
    U: size of the Aggregator's index list. A label counts as found when every holder got the
    same index and no other label shares it; union_metrics then scores the found set.
    """
    by_label, by_index = {}, {}
    for labels, index in zip(client_labels, client_index):
        for x in labels:
            by_label.setdefault(x, set()).add(index[x])
            by_index.setdefault(index[x], set()).add(x)
    split = sorted(x for x, s in by_label.items() if len(s) > 1)
    merged = sorted(k for k, s in by_index.items() if len(s) > 1)
    found = [x for x, s in by_label.items() if len(s) == 1 and len(by_index[next(iter(s))]) == 1]
    real, _ = true_union(client_labels)
    out = union_metrics(found, real, dictionary)
    out.update(index_list_size=U, split_labels=split, merged_indices=merged,
               out_of_range=sorted(k for k in by_index if not 0 <= k < U),
               unused_indices=sorted(set(range(U)) - set(by_index)))
    out['exact'] = out['exact'] and U == len(real) and not (split or merged or out['out_of_range']
                                                             or out['unused_indices'])
    if not out['mcc_defined']:
        out['mcc'] = float(out['exact'])                                # fallback follows the final verdict
    return out
