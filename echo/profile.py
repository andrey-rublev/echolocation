"""Echo profiles: matched filter, envelope, range conversion, background subtraction.

Two entry points:

* :func:`matched_filter` / :func:`envelope` operate on a whole recording and are
  what the tests and the simulator use directly.
* :class:`EchoProcessor` turns a continuous capture stream (arbitrary block
  sizes, arbitrary system latency) into one range profile per chirp period.

Range convention: zero range is the direct-path peak plus a calibrated offset
(``profile.calib_offset_samples``), so the unknown audio-stack latency cancels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, find_peaks, hilbert, lfilter

from echo.chirp import make_chirp
from echo.config import Config


# --------------------------------------------------------------------------- #
# Whole-recording primitives
# --------------------------------------------------------------------------- #
def matched_filter(recording: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-correlate ``recording`` with ``template``.

    Output index ``n`` is the correlation at lag ``n``: an echo whose chirp
    starts at sample ``n`` of the recording produces a peak at index ``n``.
    """
    full = fftconvolve(recording, template[::-1], mode="full")
    start = len(template) - 1
    return full[start : start + len(recording)]


def envelope(x: np.ndarray) -> np.ndarray:
    """Magnitude of the analytic signal (Hilbert envelope)."""
    return np.abs(hilbert(x))


def delay_to_meters(delay_samples: float | np.ndarray, sample_rate: int, c: float = 343.0):
    """Round-trip delay in samples -> one-way distance in metres."""
    return np.asarray(delay_samples) / sample_rate * c / 2.0


def meters_to_delay(meters: float | np.ndarray, sample_rate: int, c: float = 343.0):
    """One-way distance in metres -> round-trip delay in samples."""
    return np.asarray(meters) * 2.0 / c * sample_rate


def parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position by fitting a parabola through y[i-1:i+2]."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(i)
    return float(i + 0.5 * (a - c) / denom)


def find_echo_delays(
    env: np.ndarray, n: int, min_separation: int = 20, min_height: float = 0.0
) -> list[float]:
    """Return up to ``n`` strongest peak positions (sub-sample), sorted by delay."""
    peaks, props = find_peaks(env, distance=min_separation, height=min_height)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1][:n]
    return sorted(parabolic_peak(env, int(peaks[j])) for j in order)


# --------------------------------------------------------------------------- #
# Range axis helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RangeGrid:
    """Geometry of a cropped, decimated profile."""

    sample_rate: int
    speed_of_sound: float
    n_range_samples: int  # lag samples covered (before decimation)
    decimate: int

    @property
    def n_bins(self) -> int:
        return self.n_range_samples // self.decimate

    @property
    def bin_m(self) -> float:
        return float(delay_to_meters(self.decimate, self.sample_rate, self.speed_of_sound))

    def axis_m(self) -> np.ndarray:
        """Range (m) at the centre of each bin."""
        centres = (np.arange(self.n_bins) + 0.5) * self.decimate - 0.5
        return delay_to_meters(centres, self.sample_rate, self.speed_of_sound)

    def bin_of(self, meters: float) -> int:
        lag = float(meters_to_delay(meters, self.sample_rate, self.speed_of_sound))
        return int(np.clip(lag // self.decimate, 0, self.n_bins - 1))


def range_grid(cfg: Config) -> RangeGrid:
    p = cfg.profile
    fs = cfg.audio.sample_rate
    n = int(math.ceil(float(meters_to_delay(p.max_range_m, fs, p.speed_of_sound_mps))))
    n = int(math.ceil(n / p.decimate) * p.decimate)
    return RangeGrid(fs, p.speed_of_sound_mps, n, p.decimate)


# --------------------------------------------------------------------------- #
# Background subtraction
# --------------------------------------------------------------------------- #
class EmaBackground:
    """Exponential-moving-average background; ``update`` returns profile - background.

    The difference is taken against the background *before* it absorbs the
    new frame, so a sudden change shows at full strength on its first frame.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._bg: np.ndarray | None = None

    @property
    def background(self) -> np.ndarray | None:
        return None if self._bg is None else self._bg.copy()

    def reset(self) -> None:
        self._bg = None

    def update(self, profile: np.ndarray) -> np.ndarray:
        x = np.asarray(profile, dtype=np.float64)
        if self._bg is None:
            self._bg = x.copy()
        diff = x - self._bg
        self._bg += self.alpha * diff
        return diff


def ema_subtract(profiles: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised equivalent of running :class:`EmaBackground` over ``profiles`` (T, B)."""
    x = np.asarray(profiles, dtype=np.float64)
    if len(x) == 0:
        return x.copy()
    zi = ((1 - alpha) * x[0])[None, :]
    bg = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=zi)[0]
    bg_prev = np.vstack([x[:1], bg[:-1]])
    return x - bg_prev


# --------------------------------------------------------------------------- #
# Streaming processor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EchoFrame:
    """One range profile, produced once per chirp period."""

    index: int  # period number since the stream started
    time_s: float  # stream time of the direct-path arrival (sample clock + t0)
    profile: np.ndarray  # (n_bins,) envelope from 0 to max_range_m
    direct_lag: float  # measured direct-path peak, sub-sample, relative to expected
    direct_amp: float  # envelope height at the direct-path peak


@dataclass(frozen=True)
class LockInfo:
    phase: int  # direct-path arrival phase within a period (samples)
    spread: float  # peak-to-peak scatter of the warm-up observations (samples)
    n_obs: int


class EchoProcessor:
    """Stream of captured samples -> :class:`EchoFrame` per chirp period.

    The playback is strictly periodic and shares the capture clock (full-duplex
    stream), so the direct path arrives at the same phase in every period. The
    processor first locks onto that phase from ``warmup_periods`` periods, then
    cuts one lag window per period around it, so any system latency (even more
    than a period) is handled without knowing it in advance.
    """

    _MARGIN = 32  # extra lags on each side to keep Hilbert edge effects out

    def __init__(self, cfg: Config, t0: float = 0.0) -> None:
        p = cfg.profile
        self.cfg = cfg
        self.t0 = t0
        self.fs = cfg.audio.sample_rate
        self.P = cfg.period_samples
        self.template = make_chirp(cfg.chirp, self.fs)
        self._rev = self.template[::-1].copy()
        self.grid = range_grid(cfg)
        self._offset = float(p.calib_offset_samples)
        self._search = int(p.direct_search_samples)
        self._align_direct = p.align == "direct"
        if self._offset < -(p.pre_samples - self._search):
            raise ValueError(
                "calib_offset_samples is more negative than pre_samples allows; "
                "increase profile.pre_samples."
            )
        self._before = p.pre_samples + self._search + self._MARGIN
        after = self._search + max(0, math.ceil(self._offset)) + self.grid.n_range_samples
        self._after = after + self._MARGIN
        self._window = self._before + self._after  # lags per frame
        self._seg_len = self._window + len(self.template) - 1
        if self._window > self.P:
            raise ValueError("Lag window exceeds one period; shorten max_range_m.")

        self._buf = np.zeros(0, dtype=np.float64)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._warm_k = 1  # skip period 0 (playback start-up transient)
        self._obs: list[int] = []
        self._lock: LockInfo | None = None
        self._next_k = 0

    # -- public -------------------------------------------------------------
    @property
    def lock(self) -> LockInfo | None:
        return self._lock

    @property
    def n_bins(self) -> int:
        return self.grid.n_bins

    def push(self, samples: np.ndarray) -> list[EchoFrame]:
        """Append captured mono samples; return any frames that became complete."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._buf = np.concatenate([self._buf, x])
        frames: list[EchoFrame] = []
        if self._lock is None:
            self._warm_up()
        if self._lock is not None:
            frames = self._emit()
        self._trim()
        return frames

    # -- internals ------------------------------------------------------------
    def _segment(self, start_abs: int, length: int) -> np.ndarray | None:
        i = start_abs - self._buf_start
        if i < 0 or i + length > len(self._buf):
            return None
        return self._buf[i : i + length]

    def _warm_up(self) -> None:
        need = self.cfg.profile.warmup_periods
        L = self.P + len(self.template) - 1
        while self._lock is None:
            seg = self._segment(self._warm_k * self.P, L)
            if seg is None:
                return
            env = np.abs(hilbert(fftconvolve(seg, self._rev, mode="valid")))
            peak = int(np.argmax(env))
            # Accept only frames with a clear peak (rules out silence / dropouts).
            if env[peak] > 8.0 * (np.median(env) + 1e-12):
                self._obs.append(peak)
            self._warm_k += 1
            if len(self._obs) >= need:
                self._lock = self._make_lock()
                # First emitted frame: the earliest one still fully in the buffer.
                first = self._buf_start + self._before - self._lock.phase
                self._next_k = max(0, math.ceil(first / self.P))

    def _make_lock(self) -> LockInfo:
        obs = np.asarray(self._obs)
        ref = obs[0]
        d = (obs - ref + self.P // 2) % self.P - self.P // 2  # circular unwrap
        phase = int((ref + int(np.round(np.median(d)))) % self.P)
        return LockInfo(phase=phase, spread=float(np.ptp(d)), n_obs=len(obs))

    def _emit(self) -> list[EchoFrame]:
        assert self._lock is not None
        out: list[EchoFrame] = []
        while True:
            k = self._next_k
            expected = k * self.P + self._lock.phase
            seg = self._segment(expected - self._before, self._seg_len)
            if seg is None:
                break
            out.append(self._frame(k, expected, seg))
            self._next_k += 1
        return out

    def _frame(self, k: int, expected_abs: int, seg: np.ndarray) -> EchoFrame:
        env = np.abs(hilbert(fftconvolve(seg, self._rev, mode="valid")))
        lo, hi = self._before - self._search, self._before + self._search + 1
        j = lo + int(np.argmax(env[lo:hi]))
        j_sub = parabolic_peak(env, j)
        anchor = j_sub if self._align_direct else float(self._before)
        zero = anchor + self._offset
        pos = zero + np.arange(self.grid.n_range_samples)
        prof = np.interp(pos, np.arange(len(env)), env)
        if self.grid.decimate > 1:
            prof = prof.reshape(-1, self.grid.decimate).mean(axis=1)
        return EchoFrame(
            index=k,
            time_s=self.t0 + expected_abs / self.fs,
            profile=prof.astype(np.float32),
            direct_lag=j_sub - self._before,
            direct_amp=float(env[j]),
        )

    def _trim(self) -> None:
        if self._lock is None:
            keep_from = self._warm_k * self.P
        else:
            keep_from = self._next_k * self.P + self._lock.phase - self._before
        # Never drop more than we hold: samples not yet received keep their index.
        drop = min(keep_from - self._buf_start, len(self._buf))
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop


def process_recording(cfg: Config, recording: np.ndarray, t0: float = 0.0) -> list[EchoFrame]:
    """Run a whole mono recording through :class:`EchoProcessor`."""
    proc = EchoProcessor(cfg, t0=t0)
    return proc.push(recording)


def stack_profiles(frames: list[EchoFrame]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([f.profile for f in frames])
