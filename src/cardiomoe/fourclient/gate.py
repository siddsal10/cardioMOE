"""Weighted gated aggregation of the four clients' class probabilities.

    P_final(y|x) = sum_i w_i P_i(y|x),   w_i >= 0,   sum_i w_i = 1

Weights live on the simplex through a softmax, so the constraint holds by construction.
  * EqualWeights   w_i = 1/4 (reference, nothing learned)
  * StaticWeights  one global w per client, learned on validation data
  * DynamicGate    w(x) per patient, from the clients' probabilities and uncertainties

All are fitted with BCE on the mixed probabilities (multi-label, 5 superclasses).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor, nn

EPS = 1e-6


def mix(P: Tensor, w: Tensor) -> Tensor:
    """P [N, K clients, C], w [K] or [N, K] -> [N, C]."""
    if w.dim() == 1:
        w = w.expand(P.shape[0], -1)
    return (w.unsqueeze(-1) * P).sum(1)


def bce_prob(p: Tensor, y: Tensor) -> Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()


def entropy(P: Tensor) -> Tensor:
    """Mean binary entropy per client: [N, K, C] -> [N, K]."""
    p = P.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean(-1)


class EqualWeights:
    name = "Equal-weight mean"

    def fit(self, P, y, groups=None):
        self.w = torch.full((P.shape[1],), 1.0 / P.shape[1])
        return self

    def weights(self, P: Tensor) -> Tensor:
        return self.w.expand(P.shape[0], -1)

    def predict(self, P: Tensor) -> Tensor:
        return mix(P, self.w)


class StaticWeights:
    name = "Static learned weights"

    def __init__(self, lr: float = 0.05, steps: int = 500):
        self.lr, self.steps = lr, steps

    def fit(self, P: Tensor, y: Tensor, groups=None):
        theta = torch.zeros(P.shape[1], requires_grad=True)
        opt = torch.optim.Adam([theta], lr=self.lr)
        for _ in range(self.steps):  # 4 parameters, convex-ish: full-batch Adam converges quickly
            opt.zero_grad()
            loss = bce_prob(mix(P, torch.softmax(theta, 0)), y)
            loss.backward()
            opt.step()
        self.w = torch.softmax(theta.detach(), 0)
        return self

    def weights(self, P: Tensor) -> Tensor:
        return self.w.expand(P.shape[0], -1)

    def predict(self, P: Tensor) -> Tensor:
        return mix(P, self.w)


class GateNet(nn.Module):
    """[N, K, C] probabilities -> per-patient weights [N, K] (softmax over clients)."""

    def __init__(self, k: int, c: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(k * c + k, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, k))
        nn.init.zeros_(self.mlp[-1].weight)  # start at equal weights
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, P: Tensor) -> Tensor:
        x = torch.cat([P.flatten(1), entropy(P)], dim=1)
        return torch.softmax(self.mlp(x), dim=1)


class DynamicGate:
    name = "Dynamic gate (per-patient)"

    def __init__(self, hidden=32, dropout=0.2, lr=1e-3, weight_decay=1e-2, epochs=300, patience=30,
                 batch_size=128, seed=42, val_frac=0.2):
        self.kw = dict(hidden=hidden, dropout=dropout)
        self.lr, self.wd, self.epochs, self.patience = lr, weight_decay, epochs, patience
        self.bs, self.seed, self.val_frac = batch_size, seed, val_frac

    def fit(self, P: Tensor, y: Tensor, groups: np.ndarray | None = None):
        """Early stopping on a patient-grouped inner hold-out of the data it is given."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        g = groups if groups is not None else np.arange(len(P))
        ug = np.unique(g)
        held = set(rng.choice(ug, max(1, int(self.val_frac * len(ug))), replace=False).tolist())
        va = np.array([x in held for x in g])
        tr_idx, va_idx = np.where(~va)[0], np.where(va)[0]
        self.net = GateNet(P.shape[1], P.shape[2], **self.kw)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(tr_idx), max(1, len(tr_idx) // self.bs)):
                opt.zero_grad()
                loss = bce_prob(mix(P[b], self.net(P[b])), y[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = bce_prob(mix(P[va_idx], self.net(P[va_idx])), y[va_idx]).item()
            if v < best - 1e-5:
                best, bad, best_state = v, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def weights(self, P: Tensor) -> Tensor:
        return self.net(P)

    @torch.no_grad()
    def predict(self, P: Tensor) -> Tensor:
        return mix(P, self.net(P))
