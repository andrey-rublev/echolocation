"""Chirp generation: a windowed frequency sweep followed by silence."""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp as _sweep
from scipy.signal.windows import tukey

from echo.config import ChirpConfig


def make_chirp(c: ChirpConfig, sample_rate: int) -> np.ndarray:
    """Unit-peak sweep from ``f0_hz`` to ``f1_hz`` with a Tukey taper.

    The taper ramps the start and end smoothly to zero so playback has no
    clicks. This is also the matched-filter template.
    """
    n = int(round(c.duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    sweep = _sweep(t, f0=c.f0_hz, t1=c.duration_s, f1=c.f1_hz, method=c.method, phi=-90)
    out = sweep * tukey(n, alpha=c.taper)
    out /= np.max(np.abs(out))
    return out.astype(np.float64)


def make_period(c: ChirpConfig, sample_rate: int) -> np.ndarray:
    """One emission period: chirp followed by zeros (unit peak, no gain applied)."""
    period = int(round(c.period_s * sample_rate))
    out = np.zeros(period, dtype=np.float64)
    ch = make_chirp(c, sample_rate)
    out[: len(ch)] = ch
    return out


def chirp_train(c: ChirpConfig, sample_rate: int, n_periods: int, gain: float) -> np.ndarray:
    """``n_periods`` back-to-back periods scaled by ``gain`` (float32 for playback)."""
    return (np.tile(make_period(c, sample_rate), n_periods) * gain).astype(np.float32)


class PeriodicPlayer:
    """Fills arbitrary-length output blocks from a repeating period.

    Used inside the audio callback: it copies slices of a precomputed period
    (no per-call allocation in :meth:`fill_into`) and keeps its own sample
    counter. ``gain`` may be changed from another thread; a float assignment is
    atomic in CPython and takes effect at the next callback.
    """

    def __init__(self, period: np.ndarray, gain: float) -> None:
        self._period = np.asarray(period, dtype=np.float32)
        self.gain = float(gain)
        self._pos = 0
        self.samples_played = 0

    @property
    def period_samples(self) -> int:
        return len(self._period)

    def fill_into(self, out: np.ndarray) -> None:
        """Write the next ``len(out)`` samples (1-D float32 view) into ``out``."""
        n, P = len(out), len(self._period)
        g = self.gain
        written = 0
        while written < n:
            take = min(n - written, P - self._pos)
            np.multiply(self._period[self._pos : self._pos + take], g, out=out[written : written + take])
            written += take
            self._pos = (self._pos + take) % P
        self.samples_played += n

    def fill(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        self.fill_into(out)
        return out
