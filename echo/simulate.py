"""Synthetic echo generator, so the whole pipeline runs without audio hardware.

Two levels:

* :func:`simulate_recording` - one recording from a template, a list of
  ``(delay, amplitude)`` echoes, a strong direct-path peak and noise.
* :class:`SimSonar` - a continuous capture stream from a simulated room with
  static reflectors, diffuse reverb, an optional moving person and an arbitrary
  system latency. It yields blocks exactly like the real full-duplex stream.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from echo.chirp import make_chirp
from echo.config import Config

_SINC_HALF = 12  # half-width of the fractional-delay kernel (taps)


@dataclass(frozen=True)
class Echo:
    delay_samples: float
    amplitude: float


def add_impulse(h: np.ndarray, delay: float, amp: float) -> None:
    """Add a band-limited (windowed-sinc) impulse at a fractional ``delay`` into ``h``.

    Integer delays reduce exactly to a single tap.
    """
    i0 = int(math.floor(delay))
    frac = delay - i0
    if frac == 0.0:
        if 0 <= i0 < len(h):
            h[i0] += amp
        return
    taps = np.arange(-_SINC_HALF + 1, _SINC_HALF + 1)
    idx = i0 + taps
    x = taps - frac
    w = 0.5 * (1 + np.cos(np.pi * x / (_SINC_HALF + 1)))  # Hann-windowed sinc
    kern = np.sinc(x) * w
    ok = (idx >= 0) & (idx < len(h))
    np.add.at(h, idx[ok], amp * kern[ok])


def impulse_response(echoes: Sequence[Echo], length: int) -> np.ndarray:
    h = np.zeros(length, dtype=np.float64)
    for e in echoes:
        add_impulse(h, e.delay_samples, e.amplitude)
    return h


def simulate_recording(
    template: np.ndarray,
    echoes: Sequence[Echo],
    n_samples: int,
    direct: Echo | None = Echo(10.0, 1.0),
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Recording of one chirp: direct path + echoes + white noise.

    Delays are measured from the start of the recording to the start of the
    arriving chirp, which is where :func:`echo.profile.matched_filter` peaks.
    """
    rng = rng or np.random.default_rng(0)
    taps = list(echoes) + ([direct] if direct is not None else [])
    h = impulse_response(taps, n_samples)
    rec = fftconvolve(h, template, mode="full")[:n_samples]
    if noise_std > 0:
        rec = rec + rng.normal(0.0, noise_std, n_samples)
    return rec


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
Trajectory = Callable[[float], "float | None"]
"""Maps time (s) to target range (m), or None when nobody is in range."""


def absent() -> Trajectory:
    return lambda t: None


def static_at(distance_m: float) -> Trajectory:
    return lambda t: distance_m


def linear_motion(r0: float, r1: float, duration_s: float) -> Trajectory:
    """Move from r0 to r1 over duration_s, then bounce back (triangle wave)."""

    def f(t: float) -> float:
        ph = (t / duration_s) % 2.0
        frac = ph if ph <= 1.0 else 2.0 - ph
        return r0 + (r1 - r0) * frac

    return f


def hand_wave(center_m: float = 0.3, amplitude_m: float = 0.08, hz: float = 1.0) -> Trajectory:
    return lambda t: center_m + amplitude_m * math.sin(2 * math.pi * hz * t)


def random_walk(
    rng: np.random.Generator,
    r_min: float = 0.5,
    r_max: float = 3.0,
    speed_mps: float = 0.6,
    duration_s: float = 600.0,
    leave_prob: float = 0.0,
) -> Trajectory:
    """Piecewise-linear walk between random waypoints.

    With ``leave_prob`` > 0, each leg may instead be an absence (person out of
    range) of a few seconds, giving sessions with mixed presence.
    """
    times = [0.0]
    ranges: list[float | None] = [float(rng.uniform(r_min, r_max))]
    t = 0.0
    while t < duration_s:
        if leave_prob > 0 and rng.random() < leave_prob:
            dt = float(rng.uniform(2.0, 6.0))
            times += [t + 1e-3, t + dt]
            ranges += [None, None]
            t += dt
            times.append(t + 1e-3)
            ranges.append(float(rng.uniform(r_min, r_max)))
            continue
        nxt = float(rng.uniform(r_min, r_max))
        prev = next(r for r in reversed(ranges) if r is not None)
        dt = max(abs(nxt - prev) / speed_mps, 0.5) + float(rng.uniform(0.0, 2.0))
        t += dt
        times.append(t)
        ranges.append(nxt)
    tt = np.asarray(times)

    def f(time_s: float) -> float | None:
        i = int(np.searchsorted(tt, time_s, side="right")) - 1
        i = min(max(i, 0), len(tt) - 2)
        a, b = ranges[i], ranges[i + 1]
        if a is None or b is None:
            return None
        w = (time_s - tt[i]) / max(tt[i + 1] - tt[i], 1e-9)
        return a + (b - a) * min(max(w, 0.0), 1.0)

    return f


@dataclass(frozen=True)
class Room:
    """Static acoustic environment: discrete reflectors plus a diffuse tail."""

    name: str
    reflectors: tuple[tuple[float, float], ...]  # (range_m, amplitude)
    diffuse_taps: tuple[tuple[float, float], ...] = field(default=())


def make_room(name: str) -> Room:
    """Deterministic pseudo-random room layout derived from ``name``."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    refl: list[tuple[float, float]] = [(float(rng.uniform(0.25, 0.4)), 0.02)]  # desk/screen
    for _ in range(int(rng.integers(3, 6))):
        r = float(rng.uniform(0.8, 4.2))
        refl.append((r, float(rng.uniform(0.2, 0.7)) * 0.12 / (2 * r)))
    diffuse: list[tuple[float, float]] = []
    for _ in range(120):
        r = float(rng.uniform(0.5, 6.0))
        diffuse.append((r, float(rng.normal(0, 1)) * 0.004 * math.exp(-r / 1.5)))
    return Room(name, tuple(refl), tuple(diffuse))


@dataclass(frozen=True)
class PersonModel:
    """A person as a few scatterers spread in range, with breathing sway."""

    strength: float = 0.012  # amplitude at 1 m (falls off as 1/r^2)
    parts: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.06, 0.5), (-0.04, 0.35), (0.15, 0.25))
    sway_m: float = 0.004
    sway_hz: float = 0.25


class SimSonar:
    """Simulated full-duplex capture: yields consecutive mono input blocks.

    The emitted signal is a periodic chirp train. For period ``k`` the scene's
    impulse response at that instant is convolved with the chirp and lands in
    the capture stream at ``k * P + latency`` (+ optional jitter).
    """

    device_name = "simulator"

    def __init__(
        self,
        cfg: Config,
        room: Room | None = None,
        person: Trajectory | None = None,
        gain: float | None = None,
        realtime: bool = False,
        agc: bool = False,
        person_model: PersonModel = PersonModel(),
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.fs = cfg.audio.sample_rate
        self.P = cfg.period_samples
        self.room = room or make_room("sim_room")
        self.person = person or absent()
        self.gain = cfg.chirp.gain if gain is None else gain
        self.realtime = realtime
        self.agc = agc
        self.person_model = person_model
        self.rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)
        self._chirp = make_chirp(cfg.chirp, self.fs)
        self._c = cfg.profile.speed_of_sound_mps
        self._ir_len = int(meters_to_samples(7.0, self.fs, self._c))
        self._static = self._static_ir()
        # Accumulator for overlapping contributions; index 0 = absolute sample _acc_start.
        tail = self._ir_len + len(self._chirp) + cfg.sim.latency_samples + 4 * self.P
        self._acc = np.zeros(tail, dtype=np.float64)
        self._acc_start = 0
        self._k = 0
        self._agc_level = 1.0

    # -- scene ----------------------------------------------------------------
    def _static_ir(self) -> np.ndarray:
        s = self.cfg.sim
        h = np.zeros(self._ir_len)
        add_impulse(h, meters_to_samples(s.speaker_mic_m / 2, self.fs, self._c), s.direct_amplitude)
        for r, a in self.room.reflectors + self.room.diffuse_taps:
            add_impulse(h, meters_to_samples(r, self.fs, self._c), a)
        return h

    def truth(self, t: float) -> float | None:
        """Ground-truth person range at time ``t`` (None = absent)."""
        return self.person(t)

    def _ir_at(self, t: float) -> np.ndarray:
        r = self.person(t)
        if r is None:
            return self._static
        h = self._static.copy()
        pm = self.person_model
        sway = pm.sway_m * math.sin(2 * math.pi * pm.sway_hz * t)
        for dr, w in pm.parts:
            rr = max(r + dr + sway, 0.05)
            add_impulse(h, meters_to_samples(rr, self.fs, self._c), w * pm.strength / rr**2)
        return h

    # -- stream ---------------------------------------------------------------
    def _render_period(self) -> None:
        k = self._k
        t = k * self.P / self.fs
        y = fftconvolve(self._ir_at(t), self._chirp * self.gain)
        jit = self.cfg.sim.jitter_samples
        shift = int(self.rng.integers(-jit, jit + 1)) if jit > 0 else 0
        start = k * self.P + self.cfg.sim.latency_samples + shift - self._acc_start
        self._acc[start : start + len(y)] += y
        self._k += 1

    def read_block(self) -> np.ndarray:
        """Return the next ``P`` captured samples."""
        # Contributions to samples [a, a+P) come from periods up to the one
        # emitted at a+P-latency, so render far enough ahead.
        a = self._acc_start
        while self._k * self.P + self.cfg.sim.latency_samples - self.cfg.sim.jitter_samples <= a + self.P:
            self._render_period()
        block = self._acc[: self.P].copy()
        self._acc = np.concatenate([self._acc[self.P :], np.zeros(self.P)])
        self._acc_start += self.P
        noise = self.rng.normal(0.0, self.cfg.sim.noise_std, self.P)
        block += noise
        if self.agc:
            # Crude automatic gain control: normalise towards a fixed level.
            level = float(np.sqrt(np.mean(block**2))) + 1e-9
            self._agc_level = 0.9 * self._agc_level + 0.1 * (0.02 / level)
            block *= self._agc_level
        return block.astype(np.float32)

    def blocks(self, n_blocks: int | None = None) -> Iterator[np.ndarray]:
        """Yield capture blocks (each one period long); paces to real time if asked."""
        start = time.perf_counter()
        i = 0
        while n_blocks is None or i < n_blocks:
            if self.realtime:
                due = start + i * self.P / self.fs
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield self.read_block()
            i += 1

    def record(self, seconds: float) -> np.ndarray:
        n = int(math.ceil(seconds * self.fs / self.P))
        return np.concatenate(list(self.blocks(n)))


def meters_to_samples(meters: float, fs: int, c: float) -> float:
    return meters * 2.0 / c * fs
