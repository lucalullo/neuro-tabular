"""Prospective report-only metrics. No fit, threshold selection or real-data run.

Not part of the frozen P6 scorer until its metric delta is explicitly approved.
PR-AUC is reported specifically as non-interpolated average precision (AP).
"""
import math


def average_precision(y_positive, scores):
    if len(y_positive) != len(scores) or not scores:
        raise ValueError('Aligned nonempty labels/scores required')
    if any(type(value) is not bool for value in y_positive):
        raise ValueError('Explicit Boolean positive labels required')
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError('Nonfinite score')
    positives = sum(y_positive)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    seen, found, result, cursor = 0, 0, 0.0, 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        added = sum(y_positive[order[index]] for index in range(cursor, end))
        seen += end - cursor
        found += added
        result += (added / positives) * (found / seen)
        cursor = end
    return result


def classification_details(y, probabilities, classes, *, native_labels,
                           positive_label=None, reference_native_labels=None):
    classes, y, native_labels = list(classes), list(y), list(native_labels)
    rows = [list(row) for row in probabilities]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('Unique recorded class order required')
    if not y or len(rows) != len(y) or len(native_labels) != len(y):
        raise ValueError('Aligned native decisions required; never guess argmax at ties')
    lookup = {value: index for index, value in enumerate(classes)}
    if any(label not in lookup for label in y + native_labels):
        raise ValueError('Unknown class')
    for row in rows:
        if len(row) != len(classes) or any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in row):
            raise ValueError('Invalid probability columns')
        if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError('Probabilities must already be normalized')
    confusion = [[0 for _ in classes] for _ in classes]
    for truth, predicted in zip(y, native_labels, strict=True):
        confusion[lookup[truth]][lookup[predicted]] += 1
    per_class = []
    for index, label in enumerate(classes):
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        true_positive = confusion[index][index]
        per_class.append(dict(label=label, support=support, predicted=predicted,
            precision=true_positive / predicted if predicted else 0.0,
            recall=true_positive / support if support else 0.0,
            absent_true_class=support == 0))
    result = dict(classes=classes, confusion=confusion, per_class=per_class,
                  decisions_source='RECORDED_NATIVE_PREDICT_NO_NEW_THRESHOLD',
                  average_precision=None, average_precision_reason='Multiclass AP not part of this proposal')
    if len(classes) == 2:
        positive = classes[1] if positive_label is None else positive_label
        if positive not in lookup:
            raise ValueError('Positive class absent from recorded classes')
        values = average_precision([label == positive for label in y], [row[lookup[positive]] for row in rows])
        result.update(positive_class=positive, average_precision=values,
                      average_precision_reason='No positive evaluation labels' if values is None else None,
                      PR_definition='NON_INTERPOLATED_AVERAGE_PRECISION_TIES_AS_ONE_THRESHOLD')
    if reference_native_labels is not None:
        reference = list(reference_native_labels)
        if len(reference) != len(y) or any(label not in lookup for label in reference):
            raise ValueError('Aligned same-TEST reference decisions required')
        result['changed_decisions'] = sum(a != b for a, b in zip(native_labels, reference, strict=True))
        result['changed_by_true_class'] = [sum(t == label and a != b for t, a, b in zip(y, native_labels, reference, strict=True)) for label in classes]
    return result
