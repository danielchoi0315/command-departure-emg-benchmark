from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

@dataclass
class PUModel:
    name: str
    clf: object

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PUModel":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(X)
            return p
        # fallback: hard predictions -> one-hot
        yhat = self.clf.predict(X)
        K = int(np.max(yhat)) + 1
        p = np.zeros((len(yhat), K), dtype=float)
        p[np.arange(len(yhat)), yhat] = 1.0
        return p

def lda_model() -> PUModel:
    return PUModel(name="lda", clf=LinearDiscriminantAnalysis())

def logreg_model(C: float = 1.0) -> PUModel:
    return PUModel(
        name="logreg",
        clf=LogisticRegression(
            C=C,
            max_iter=2000,
            multi_class="auto",
            n_jobs=-1,
        ),
    )
