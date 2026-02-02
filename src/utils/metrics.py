import numpy as np
from collections import Counter

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = len(y_true)
    s = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        most = Counter(y_true[idx]).most_common(1)[0][1]
        s += most
    return s / total