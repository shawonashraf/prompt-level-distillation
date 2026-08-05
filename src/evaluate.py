from sklearn.metrics import f1_score


def compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> float:
    valid = [
        (t, p) for t, p in zip(y_true, y_pred) if p and t in labels
    ]
    if not valid:
        return 0.0

    true_labels, pred_labels = zip(*valid)
    return f1_score(
        true_labels,
        pred_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
