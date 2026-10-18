"""The four client architectures. All take batch["signal"] [B, 12, 1000] and return 5 logits.

    A  cnn1d     plain 1D CNN (4 conv blocks + global pooling)
    B  resnet1d  1D ResNet (xresnet50-style depth)
    C  bilstm    2-layer bidirectional LSTM over the raw time series
    D  cnn_lstm  conv front end (8x downsampling) feeding a bidirectional LSTM
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from cardiomoe.clients.base import BaseClient


class ResBlock1d(nn.Module):
    """Basic pre-activation-free residual block (two convs), xresnet-style shortcut."""

    def __init__(self, c_in: int, c_out: int, k: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
        )
        nn.init.zeros_(self.body[-1].weight)  # zero-init last BN: block starts as identity
        short = []
        if stride > 1:
            short.append(nn.AvgPool1d(stride, ceil_mode=True))
        if c_in != c_out:
            short += [nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out)]
        self.short = nn.Sequential(*short)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.body(x) + self.short(x))


class ECGEncoder(nn.Module):
    """1D ResNet: stem k=7/64 ch, 4 stages [64,128,256,256], avg+max pool -> 512 -> 256."""

    def __init__(self, stem: int = 64, stages=(64, 128, 256, 256), blocks=(3, 4, 6, 3), k: int = 5,
                 out_dim: int = 256, in_ch: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, c = [], stem
        for i, (ch, n) in enumerate(zip(stages, blocks)):
            for j in range(n):
                layers.append(ResBlock1d(c, ch, k, stride=2 if (j == 0 and i > 0) else 1))
                c = ch
        self.stages = nn.Sequential(*layers)
        self.proj = nn.Linear(2 * c, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stages(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.proj(h)


def conv_block(c_in: int, c_out: int, k: int = 7, pool: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool1d(pool))


def avg_max(h: Tensor, dim: int) -> Tensor:
    return torch.cat([h.mean(dim), h.amax(dim)], dim=1)


class CNN1D(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, c = [], 12
        for ch in channels:
            layers.append(conv_block(c, ch))
            c = ch
        self.body = nn.Sequential(*layers)
        self.out_dim = 2 * c

    def forward(self, x: Tensor) -> Tensor:
        return avg_max(self.body(x), -1)


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ECGEncoder(out_dim=256)
        self.out_dim = 256

    def forward(self, x: Tensor) -> Tensor:
        return self.enc(x)


class BiLSTM(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(12, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.out_dim = 4 * hidden

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(x.transpose(1, 2))  # [B, T, 2H]
        return avg_max(h, 1)


class CNNLSTM(nn.Module):
    def __init__(self, channels=(32, 64, 128), hidden: int = 128):
        super().__init__()
        layers, c = [], 12
        for ch in channels:
            layers.append(conv_block(c, ch))
            c = ch
        self.conv = nn.Sequential(*layers)  # 1000 -> 125 time steps
        self.lstm = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
        self.out_dim = 4 * hidden

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(self.conv(x).transpose(1, 2))
        return avg_max(h, 1)


ARCHS = {"cnn1d": CNN1D, "resnet1d": ResNet1D, "bilstm": BiLSTM, "cnn_lstm": CNNLSTM}
ARCH_NAMES = {"cnn1d": "1D CNN", "resnet1d": "1D ResNet", "bilstm": "BiLSTM", "cnn_lstm": "CNN-LSTM"}


class SignalClient(BaseClient):
    modality = "ecg"

    def __init__(self, arch: str, embed_dim: int = 256, dropout: float = 0.3, n_classes: int = 5):
        super().__init__()
        self.arch, self.embed_dim = arch, embed_dim
        self.backbone = ARCHS[arch]()
        self.proj = nn.Sequential(nn.Linear(self.backbone.out_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, n_classes)

    def encode(self, batch: dict) -> Tensor:
        return self.proj(self.backbone(batch["signal"]))
