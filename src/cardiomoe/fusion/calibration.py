"""Per-client temperature scaling (Section 7.3): one scalar T_i fitted on fold 9 by minimising BCE."""
from __future__ import annotations

import torch

from cardiomoe.clients.base import masked_bce


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200) -> float:
    logits, y = logits.float().detach(), y.float()
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = masked_bce(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))
