"""Sessions on disk, session-level splits, model features and the torch dataset.

A *session* is one continuous recording: ``<id>.npz`` holds per-frame arrays
(raw echo profiles, timestamps, labels) and ``<id>.json`` the metadata (room,
device, gain, date, notes, label source, config snapshot).

Frames from one session are near-duplicates of each other, so every split in
this project is by session (and by day when there are several days). Rows are
never shuffled across the train/test boundary.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from echo.config import AugmentConfig
from echo.profile import RangeGrid, ema_subtract

FEATURE_CHANNELS = ("log_profile", "log_change_slow", "log_change_fast")
SLOW_ALPHA = 0.01  # ~5 s memory: keeps a still person visible
FAST_ALPHA = 0.2  # ~0.25 s memory: highlights motion
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
_DIST_RE = re.compile(r"^(?:person|human|me)[_\-]?(?:at[_\-]?)?(\d+(?:\.\d+)?)(m|cm)$")


def parse_label(label: str) -> tuple[float, float]:
    """Manual session label -> (presence, distance_m); NaN means unknown.

    ``empty`` -> (0, nan); ``person`` / ``walking`` -> (1, nan);
    ``person_1m`` / ``person_1.5m`` / ``person_150cm`` -> (1, 1.5).
    Anything else -> (nan, nan): e.g. a room-only session.
    """
    s = label.strip().lower()
    if s in ("empty", "none", "nobody", "absent"):
        return 0.0, math.nan
    if s in ("person", "present", "walking", "walk", "moving", "sitting", "standing"):
        return 1.0, math.nan
    m = _DIST_RE.match(s)
    if m:
        d = float(m.group(1)) / (100.0 if m.group(2) == "cm" else 1.0)
        return 1.0, d
    return math.nan, math.nan


# --------------------------------------------------------------------------- #
# Session I/O
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    path: Path
    profiles: np.ndarray  # (T, B) float32 raw envelope
    times: np.ndarray  # (T,) float64, seconds (monotonic clock or stream time)
    direct_amp: np.ndarray  # (T,)
    direct_lag: np.ndarray  # (T,)
    presence: np.ndarray  # (T,) 1 / 0 / nan
    distance: np.ndarray  # (T,) metres or nan
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.meta.get("session_id", self.path.stem))

    @property
    def room(self) -> str:
        return str(self.meta.get("room", "unknown"))

    @property
    def day(self) -> str:
        return str(self.meta.get("day", str(self.meta.get("date", ""))[:10]))

    @property
    def grid(self) -> RangeGrid:
        g = self.meta["grid"]
        return RangeGrid(int(g["sample_rate"]), float(g["speed_of_sound"]),
                         int(g["n_range_samples"]), int(g["decimate"]))

    def __len__(self) -> int:
        return len(self.times)


def save_session(
    out_dir: Path,
    session_id: str,
    profiles: np.ndarray,
    times: np.ndarray,
    direct_amp: np.ndarray,
    direct_lag: np.ndarray,
    presence: np.ndarray,
    distance: np.ndarray,
    meta: dict[str, Any],
    grid: RangeGrid,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{session_id}.npz"
    np.savez_compressed(
        npz,
        profiles=np.asarray(profiles, np.float32),
        times=np.asarray(times, np.float64),
        direct_amp=np.asarray(direct_amp, np.float32),
        direct_lag=np.asarray(direct_lag, np.float32),
        presence=np.asarray(presence, np.float32),
        distance=np.asarray(distance, np.float32),
    )
    meta = dict(meta)
    meta["session_id"] = session_id
    meta["n_frames"] = int(len(times))
    meta["grid"] = {"sample_rate": grid.sample_rate, "speed_of_sound": grid.speed_of_sound,
                    "n_range_samples": grid.n_range_samples, "decimate": grid.decimate,
                    "n_bins": grid.n_bins, "bin_m": grid.bin_m}
    (out_dir / f"{session_id}.json").write_text(json.dumps(meta, indent=2, default=str),
                                                encoding="utf-8")
    return npz


def load_session(path: str | Path) -> Session:
    p = Path(path)
    npz, js = p.with_suffix(".npz"), p.with_suffix(".json")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    return Session(npz, arrays["profiles"], arrays["times"], arrays["direct_amp"],
                   arrays["direct_lag"], arrays["presence"], arrays["distance"], meta)


def list_sessions(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npz"))


def load_sessions(directory: str | Path) -> list[Session]:
    return [load_session(p) for p in list_sessions(directory)]


# --------------------------------------------------------------------------- #
# Session-level split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    group_by: str

    def to_json(self) -> dict[str, Any]:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test),
                "group_by": self.group_by}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Split:
        return Split(tuple(d["train"]), tuple(d["val"]), tuple(d["test"]), d["group_by"])


def _strata(s: Session) -> str:
    """Coarse session type used to keep every split varied."""
    pres = s.presence[np.isfinite(s.presence)]
    kind = "unlabelled" if len(pres) == 0 else ("mixed" if 0 < pres.mean() < 1
                                                 else ("person" if pres.mean() == 1 else "empty"))
    return f"{s.room}|{kind}"


def split_sessions(
    sessions: Sequence[Session],
    test_fraction: float = 0.3,
    val_fraction: float = 0.2,
    seed: int = 0,
    by_day: bool = True,
) -> Split:
    """Assign whole sessions (or whole days) to train / val / test.

    Groups are days when ``by_day`` and there are at least three distinct days,
    otherwise individual sessions. Within each room/presence stratum, groups are
    dealt round-robin in a seeded random order so each split sees every kind of
    session when there are enough of them.
    """
    days = {s.day for s in sessions}
    group_by = "day" if by_day and len(days) >= 3 else "session"
    key = (lambda s: s.day) if group_by == "day" else (lambda s: s.id)
    groups: dict[str, list[Session]] = {}
    for s in sessions:
        groups.setdefault(key(s), []).append(s)
    rng = np.random.default_rng(seed)
    names = sorted(groups)
    rng.shuffle(names)
    n = len(names)
    if n < 2:
        raise ValueError("Need at least two sessions (or days) to make a held-out split.")
    if group_by == "session":
        # Stratify: order groups so each stratum alternates between splits.
        by_stratum: dict[str, list[str]] = {}
        for g in names:
            by_stratum.setdefault(_strata(groups[g][0]), []).append(g)
        test, val, train = [], [], []
        for members in by_stratum.values():
            k = len(members)
            n_test = int(round(k * test_fraction)) if k >= 2 else 0
            n_val = int(round((k - n_test) * val_fraction)) if k - n_test >= 3 else 0
            test += members[:n_test]
            val += members[n_test:n_test + n_val]
            train += members[n_test + n_val:]
        if not test:  # tiny datasets: hold out at least one group
            test = [train.pop()]
    else:
        n_test = max(1, int(round(n * test_fraction)))
        n_val = int(round((n - n_test) * val_fraction)) if n - n_test >= 3 else 0
        test, val, train = names[:n_test], names[n_test:n_test + n_val], names[n_test + n_val:]

    def ids(gs: list[str]) -> tuple[str, ...]:
        return tuple(sorted(s.id for g in gs for s in groups[g]))

    return Split(ids(train), ids(val), ids(test), group_by)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def session_features(s: Session) -> np.ndarray:
    """Causal per-frame features, (T, C, B).

    All channels are normalised by the per-frame direct-path amplitude, which
    removes the effect of playback gain and mic sensitivity, then log-compressed:

    0. the echo profile itself (static scene, a still person),
    1. change against a slow background (someone who arrived and stays),
    2. change against a fast background (motion).

    Only past frames are used, so live inference computes exactly the same thing.
    """
    ref = np.maximum(s.direct_amp.astype(np.float64), _EPS)[:, None]
    p = s.profiles.astype(np.float64) / ref
    slow = np.abs(ema_subtract(p, SLOW_ALPHA))
    fast = np.abs(ema_subtract(p, FAST_ALPHA))
    feats = np.stack([np.log10(p + _EPS), np.log10(slow + _EPS), np.log10(fast + _EPS)], axis=1)
    return feats.astype(np.float32)


class StreamingFeatures:
    """Frame-by-frame equivalent of :func:`session_features` for live inference."""

    def __init__(self) -> None:
        self._slow: np.ndarray | None = None
        self._fast: np.ndarray | None = None

    def update(self, profile: np.ndarray, direct_amp: float) -> np.ndarray:
        p = profile.astype(np.float64) / max(direct_amp, _EPS)
        if self._slow is None or self._fast is None:
            self._slow, self._fast = p.copy(), p.copy()
        slow, fast = np.abs(p - self._slow), np.abs(p - self._fast)
        self._slow += SLOW_ALPHA * (p - self._slow)
        self._fast += FAST_ALPHA * (p - self._fast)
        out = np.stack([np.log10(p + _EPS), np.log10(slow + _EPS), np.log10(fast + _EPS)])
        return out.astype(np.float32)


@dataclass
class Normalizer:
    """Per-channel standardisation fitted on training frames only."""

    mean: np.ndarray  # (C, 1)
    std: np.ndarray  # (C, 1)

    @staticmethod
    def fit(x: np.ndarray) -> Normalizer:
        mean = x.mean(axis=(0, 2), keepdims=True)[0]
        std = x.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        return Normalizer(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}

    @staticmethod
    def from_json(d: dict[str, list[float]]) -> Normalizer:
        return Normalizer(np.asarray(d["mean"], np.float32)[:, None],
                          np.asarray(d["std"], np.float32)[:, None])


@dataclass
class FrameSet:
    """Concatenated frames of several sessions, ready for the models."""

    x: np.ndarray  # (N, C, B) features (un-normalised)
    presence: np.ndarray  # (N,) 1/0/nan
    distance: np.ndarray  # (N,) metres/nan
    room: np.ndarray  # (N,) int index into rooms, -1 unknown
    session: np.ndarray  # (N,) int index into session_ids
    session_ids: tuple[str, ...]
    rooms: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.presence)


def build_frameset(sessions: Sequence[Session], rooms: Sequence[str], skip_frames: int = 20) -> FrameSet:
    """Features and labels for ``sessions``; the first ``skip_frames`` of each
    session are dropped while the background estimates settle."""
    xs, pres, dist, room, sess = [], [], [], [], []
    room_index = {r: i for i, r in enumerate(rooms)}
    for i, s in enumerate(sessions):
        f = session_features(s)[skip_frames:]
        n = len(f)
        xs.append(f)
        pres.append(s.presence[skip_frames:])
        dist.append(s.distance[skip_frames:])
        room.append(np.full(n, room_index.get(s.room, -1)))
        sess.append(np.full(n, i))
    if not xs:
        raise ValueError("No sessions.")
    return FrameSet(np.concatenate(xs), np.concatenate(pres).astype(np.float32),
                    np.concatenate(dist).astype(np.float32), np.concatenate(room).astype(np.int64),
                    np.concatenate(sess).astype(np.int64), tuple(s.id for s in sessions),
                    tuple(rooms))


def augment(x: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator,
            norm: Normalizer) -> np.ndarray:
    """Augment a batch of *normalised* features (N, C, B).

    * gain: a residual level change of g dB (one the direct-path normalisation
      does not cancel, e.g. a speaker-only change) shifts every log10 channel
      by g/20, i.e. by g/20/std in standardised units.
    * delay jitter: shift all channels by a few range bins (edge-padded).
    * noise: additive Gaussian, in standardised units.
    """
    n = len(x)
    out = x.copy()
    if cfg.jitter_bins > 0:
        shifts = rng.integers(-cfg.jitter_bins, cfg.jitter_bins + 1, size=n)
        for i in np.flatnonzero(shifts):
            out[i] = np.roll(out[i], int(shifts[i]), axis=-1)
            if shifts[i] > 0:
                out[i, :, : shifts[i]] = out[i, :, shifts[i] : shifts[i] + 1]
            else:
                out[i, :, shifts[i]:] = out[i, :, shifts[i] - 1 : shifts[i]]
    if cfg.gain_db > 0:
        g = rng.uniform(-cfg.gain_db, cfg.gain_db, size=(n, 1, 1)) / 20.0
        out = out + g / norm.std[None]
    if cfg.noise_std > 0:
        out = out + rng.normal(0.0, cfg.noise_std, size=out.shape)
    return out.astype(np.float32)
