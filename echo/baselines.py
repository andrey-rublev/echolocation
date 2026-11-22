"""Baselines the CNN has to beat.

(a) :class:`EnergyDetector` - presence from how much the profile changed
    against its background, thresholded. No learning beyond one threshold.
(b) :class:`LinearBaseline` - logistic regression (presence, room) and ridge
    regression (distance) on the range-pooled raw profile features.

Both are fitted on training sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from echo.dataset import FrameSet
from echo.metrics import best_threshold

MIN_RANGE_M = 0.3  # ignore the direct-path skirt


def _causal_smooth(x: np.ndarray, session: np.ndarray, n: int) -> np.ndarray:
    """Causal moving average of length n that restarts at session boundaries."""
    out = np.empty_like(x, dtype=np.float64)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        c = np.cumsum(np.insert(x[idx].astype(np.float64), 0, 0.0))
        k = np.arange(1, len(idx) + 1)
        lo = np.maximum(k - n, 0)
        out[idx] = (c[k] - c[lo]) / (k - lo)
    return out


@dataclass
class EnergyDetector:
    """Presence = mean change energy beyond ``MIN_RANGE_M`` above a threshold."""

    bin_m: float
    smooth_frames: int = 5
    threshold: float = 0.0

    def score(self, fs: FrameSet) -> np.ndarray:
        start = int(np.ceil(MIN_RANGE_M / self.bin_m))
        # Channels 1 and 2 are log10 change vs slow / fast background.
        energy = np.mean(10.0 ** fs.x[:, 1, start:], axis=1) + np.mean(10.0 ** fs.x[:, 2, start:], axis=1)
        return _causal_smooth(np.log10(energy + 1e-9), fs.session, self.smooth_frames)

    def fit(self, fs: FrameSet) -> EnergyDetector:
        self.threshold = best_threshold(fs.presence, self.score(fs))
        return self

    def predict_proba(self, fs: FrameSet) -> np.ndarray:
        """Not a calibrated probability: a logistic squash around the threshold."""
        return 1.0 / (1.0 + np.exp(-4.0 * (self.score(fs) - self.threshold)))


def pooled_features(x: np.ndarray, pool: int = 4) -> np.ndarray:
    """(N, C, B) -> (N, C * B // pool) by average-pooling the range axis."""
    n, c, b = x.shape
    b2 = (b // pool) * pool
    return x[:, :, :b2].reshape(n, c, b2 // pool, pool).mean(axis=3).reshape(n, -1)


def _fit_logistic(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-2,
                  iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial (or binary when n_classes == 1) logistic regression, full-batch L-BFGS."""
    xt = torch.from_numpy(x.astype(np.float32))
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    if n_classes == 1:
        yt = torch.from_numpy(y.astype(np.float32))
        pos = float(yt.mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-3))
        loss_fn = lambda z: torch.nn.functional.binary_cross_entropy_with_logits(  # noqa: E731
            z[:, 0], yt, pos_weight=pw)
    else:
        yt = torch.from_numpy(y.astype(np.int64))
        loss_fn = lambda z: torch.nn.functional.cross_entropy(z, yt)  # noqa: E731

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(xt @ w + b) + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


@dataclass
class LinearBaseline:
    pool: int = 4
    l2: float = 1e-2
    ridge: float = 10.0

    def __post_init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.w_p: tuple[np.ndarray, np.ndarray] | None = None
        self.w_d: np.ndarray | None = None
        self.w_r: tuple[np.ndarray, np.ndarray] | None = None
        self.n_rooms = 0

    def _feat(self, fs: FrameSet) -> np.ndarray:
        f = pooled_features(fs.x, self.pool)
        assert self.mu is not None and self.sd is not None
        return (f - self.mu) / self.sd

    def fit(self, fs: FrameSet) -> LinearBaseline:
        f = pooled_features(fs.x, self.pool)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-6
        f = (f - self.mu) / self.sd
        ok = np.isfinite(fs.presence)
        if ok.any() and 0 < fs.presence[ok].mean() < 1:
            self.w_p = _fit_logistic(f[ok], fs.presence[ok], 1, self.l2)
        dok = np.isfinite(fs.distance) & (fs.presence == 1)
        if dok.sum() > f.shape[1] // 4:
            a = np.hstack([f[dok], np.ones((dok.sum(), 1))])
            reg = self.ridge * np.eye(a.shape[1])
            reg[-1, -1] = 0.0
            self.w_d = np.linalg.solve(a.T @ a + reg, a.T @ fs.distance[dok])
        self.n_rooms = len(fs.rooms)
        rok = fs.room >= 0
        if self.n_rooms >= 2 and rok.any():
            self.w_r = _fit_logistic(f[rok], fs.room[rok], self.n_rooms, self.l2)
        return self

    def predict(self, fs: FrameSet) -> dict[str, np.ndarray]:
        f = self._feat(fs)
        n = len(f)
        out: dict[str, np.ndarray] = {
            "presence": np.full(n, np.nan),
            "distance": np.full(n, np.nan),
            "room": np.full(n, -1, dtype=np.int64),
        }
        if self.w_p is not None:
            z = f @ self.w_p[0] + self.w_p[1]
            out["presence"] = 1.0 / (1.0 + np.exp(-z[:, 0]))
        if self.w_d is not None:
            out["distance"] = np.hstack([f, np.ones((n, 1))]) @ self.w_d
        if self.w_r is not None:
            out["room"] = np.argmax(f @ self.w_r[0] + self.w_r[1], axis=1)
        return out
