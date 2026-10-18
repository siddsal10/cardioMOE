"""Common client interface (Section 6.1). The gate only ever sees ClientOutput."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ClientOutput:
    embedding: Tensor    # [B, d_i] penultimate representation
    logits: Tensor       # [B, 5]   raw (uncalibrated) logits
    uncertainty: Tensor  # [B]      mean binary entropy of sigmoid(logits / T)


def binary_entropy(logits: Tensor) -> Tensor:
    """Mean over classes of the binary entropy (nats) of sigmoid(logits)."""
    p = torch.sigmoid(logits)
    eps = 1e-7
    h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return h.mean(dim=-1)


class BaseClient(nn.Module):
    modality: str = ""
    embed_dim: int = 256

    def __init__(self) -> None:
        super().__init__()
        # Fitted on fold 9 after training (Section 7.3); 1.0 until then.
        self.register_buffer("temperature", torch.ones(()))

    def encode(self, batch: dict) -> Tensor:
        raise NotImplementedError

    def forward(self, batch: dict) -> ClientOutput:
        emb = self.encode(batch)
        logits = self.head(emb)
        return ClientOutput(emb, logits, binary_entropy(logits / self.temperature))


def masked_bce(logits: Tensor, y: Tensor, pos_weight: Tensor | None = None) -> Tensor:
    """BCE over entries with a known label (y >= 0); unknown labels are -1."""
    known = y >= 0
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y.clamp(min=0), pos_weight=pos_weight, reduction="none")
    return (loss * known).sum() / known.sum().clamp(min=1)
