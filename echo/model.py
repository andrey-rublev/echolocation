"""EchoNet: a small 1D CNN over range profiles with three heads.

Input: (N, 3, B) standardised features (profile, slow change, fast change).

* presence - global mean+max pooling -> MLP -> 1 logit
* distance - a per-position "where is the person" map at 1/4 range
  resolution; softmax over range, then the expected range (soft-argmax). This
  keeps the distance head translation-equivariant and bounded to the range
  window, which suits a tiny model and small datasets.
* room     - global pooling -> MLP -> n_rooms logits (only when >= 2 rooms)

About 25k parameters for 280 range bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _block(cin: int, cout: int, k: int, dilation: int = 1) -> nn.Sequential:
    pad = dilation * (k - 1) // 2
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    )


@dataclass(frozen=True)
class EchoNetConfig:
    n_bins: int
    bin_m: float
    in_channels: int = 3
    n_rooms: int = 0
    width: int = 32
    dropout: float = 0.1


class EchoNet(nn.Module):
    def __init__(self, cfg: EchoNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.features = nn.Sequential(
            _block(cfg.in_channels, w // 2, 7),
            _block(w // 2, w * 3 // 4, 5),
            nn.MaxPool1d(2),
            _block(w * 3 // 4, w, 5),
            nn.MaxPool1d(2),
            _block(w, w * 3 // 2, 5, dilation=2),
            _block(w * 3 // 2, w * 3 // 2, 3),
        )
        c = w * 3 // 2
        self.range_map = nn.Conv1d(c, 1, 1)
        self.presence = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * c, 32),
                                      nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.room = (nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * c, 32),
                                   nn.ReLU(inplace=True), nn.Linear(32, cfg.n_rooms))
                     if cfg.n_rooms >= 2 else None)
        # Centre (m) of each position in the 4x-downsampled range map.
        n4 = cfg.n_bins // 4
        centres = (torch.arange(n4, dtype=torch.float32) * 4 + 1.5 + 0.5) * cfg.bin_m
        self.register_buffer("centres", centres, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.features(x)  # (N, C, B/4)
        n4 = self.centres.shape[0]
        h = h[:, :, :n4]
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        range_logits = self.range_map(h)[:, 0]  # (N, B/4)
        weights = torch.softmax(range_logits, dim=1)
        out = {
            "presence": self.presence(pooled)[:, 0],
            "distance": (weights * self.centres).sum(dim=1),
            "range_logits": range_logits,
        }
        if self.room is not None:
            out["room"] = self.room(pooled)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
