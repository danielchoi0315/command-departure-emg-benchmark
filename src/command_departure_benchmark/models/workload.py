from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

class WorkloadModel:
    """Simple workload classifier (binary) returning calibrated-ish probabilities."""
    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WorkloadModel":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]  # P(high workload)
