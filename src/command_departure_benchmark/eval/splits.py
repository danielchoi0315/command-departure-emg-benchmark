from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, List
import numpy as np

@dataclass(frozen=True)
class SubjectKFold:
    n_splits: int = 5
    seed: int = 0

    def split(self, subject_ids: List[str]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) indices for subject-wise splits."""
        rng = np.random.default_rng(self.seed)
        subs = np.array(subject_ids)
        uniq = np.unique(subs)
        rng.shuffle(uniq)

        folds = np.array_split(uniq, self.n_splits)
        for k in range(self.n_splits):
            test_subs = set(folds[k].tolist())
            test_idx = np.where(np.isin(subs, list(test_subs)))[0]
            train_idx = np.where(~np.isin(subs, list(test_subs)))[0]
            yield train_idx, test_idx
