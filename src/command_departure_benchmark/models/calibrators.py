from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class TempScaler:
    """Temperature scaling for probability calibration.

    Fit on logits + labels, then transform logits -> calibrated probs.
    """
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 2000, lr: float = 0.01) -> "TempScaler":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        T = torch.nn.Parameter(torch.ones((), device=device) * float(self.temperature))
        nll = nn.CrossEntropyLoss()
        opt = optim.LBFGS([T], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(logits_t / T, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(T.detach().cpu().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        T = float(self.temperature)
        z = logits / max(T, 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
