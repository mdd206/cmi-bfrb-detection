from sklearn.metrics import f1_score

BFRB_GESTURES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]


def competition_score(y_true, y_pred):
    """Average of binary F1 (BFRB vs non-BFRB) and macro F1 over gestures."""
    true_bin = [g in BFRB_GESTURES for g in y_true]
    pred_bin = [g in BFRB_GESTURES for g in y_pred]
    binary_f1 = f1_score(true_bin, pred_bin, zero_division=0)

    # All non-BFRB gestures count as one class
    true_merged = [g if g in BFRB_GESTURES else "non_target" for g in y_true]
    pred_merged = [g if g in BFRB_GESTURES else "non_target" for g in y_pred]
    macro_f1 = f1_score(true_merged, pred_merged, average="macro", zero_division=0)

    return (binary_f1 + macro_f1) / 2
