"""Typed configuration loaded from ``echo/config.yaml``.

Every module receives a :class:`Config` (or one of its sections) explicitly;
there is no module-level mutable state.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Absolute ceiling enforced in code regardless of config: digital full scale.
FULL_SCALE = 1.0


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 48000
    input_device: str | int | None = None
    output_device: str | int | None = None
    host_api: str = "Windows WASAPI"
    wasapi_exclusive: bool = False
    output_channels: int = 2
    input_channels: int = 1
    input_channel: int = 0
    latency: str | float = "low"


@dataclass(frozen=True)
class ChirpConfig:
    f0_hz: float = 2000.0
    f1_hz: float = 18000.0
    duration_s: float = 0.020
    period_s: float = 0.050
    method: str = "linear"
    taper: float = 0.25
    gain: float = 0.1


@dataclass(frozen=True)
class SafetyConfig:
    max_gain: float = 0.3


@dataclass(frozen=True)
class ProfileConfig:
    speed_of_sound_mps: float = 343.0
    max_range_m: float = 4.0
    decimate: int = 4
    pre_samples: int = 64
    direct_search_samples: int = 24
    align: str = "direct"
    warmup_periods: int = 10
    calib_offset_samples: float = 0.0
    background_alpha: float = 0.05


@dataclass(frozen=True)
class SimConfig:
    seed: int = 0
    latency_samples: int = 3517
    speaker_mic_m: float = 0.12
    direct_amplitude: float = 1.0
    noise_std: float = 0.002
    jitter_samples: int = 0


@dataclass(frozen=True)
class CollectConfig:
    out_dir: str = "data/sessions"
    save_wav: bool = False


@dataclass(frozen=True)
class CameraConfig:
    device: int = 0
    model_path: str = "models/pose_landmarker_lite.task"
    shoulder_k_px_m: float = 230.0
    torso_to_shoulder: float = 1.3
    min_visibility: float = 0.5
    max_label_age_s: float = 0.25


@dataclass(frozen=True)
class AugmentConfig:
    noise_std: float = 0.05
    gain_db: float = 2.0
    jitter_bins: int = 2


@dataclass(frozen=True)
class TrainConfig:
    out_dir: str = "runs"
    num_threads: int = 4
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.002
    weight_decay: float = 0.0001
    test_fraction: float = 0.3
    val_fraction: float = 0.2
    patience: int = 8
    seed: int = 0
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    chirp: ChirpConfig = field(default_factory=ChirpConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def period_samples(self) -> int:
        return int(round(self.chirp.period_s * self.audio.sample_rate))

    @property
    def chirp_samples(self) -> int:
        return int(round(self.chirp.duration_s * self.audio.sample_rate))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any] | None, where: str) -> T:
    """Instantiate a (possibly nested) dataclass, rejecting unknown keys."""
    data = data or {}
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown config key(s) in [{where}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        sub = _DATACLASS_FIELDS.get((cls, name))
        if sub is not None:
            kwargs[name] = _build(sub, value, f"{where}.{name}" if where else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_DATACLASS_FIELDS: dict[tuple[type, str], type] = {
    (Config, "audio"): AudioConfig,
    (Config, "chirp"): ChirpConfig,
    (Config, "safety"): SafetyConfig,
    (Config, "profile"): ProfileConfig,
    (Config, "sim"): SimConfig,
    (Config, "collect"): CollectConfig,
    (Config, "camera"): CameraConfig,
    (Config, "train"): TrainConfig,
    (TrainConfig, "augment"): AugmentConfig,
}


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config (defaults to ``echo/config.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data)


def validate(cfg: Config) -> None:
    c = cfg.chirp
    fs = cfg.audio.sample_rate
    if not 0 < c.f0_hz < fs / 2 or not 0 < c.f1_hz < fs / 2:
        raise ValueError("Chirp frequencies must be between 0 and Nyquist.")
    if c.duration_s >= c.period_s:
        raise ValueError("chirp.duration_s must be shorter than chirp.period_s.")
    if c.method not in ("linear", "logarithmic"):
        raise ValueError("chirp.method must be 'linear' or 'logarithmic'.")
    if not 0.0 <= c.taper <= 1.0:
        raise ValueError("chirp.taper must be in [0, 1].")
    if cfg.profile.align not in ("direct", "fixed"):
        raise ValueError("profile.align must be 'direct' or 'fixed'.")
    if cfg.profile.decimate < 1:
        raise ValueError("profile.decimate must be >= 1.")
    if not 0 < cfg.safety.max_gain <= FULL_SCALE:
        raise ValueError("safety.max_gain must be in (0, 1].")
    # The range window plus the chirp must fit in one period, otherwise echoes
    # of consecutive chirps overlap inside a single profile.
    max_lag = 2 * cfg.profile.max_range_m / cfg.profile.speed_of_sound_mps * fs
    needed = max_lag + cfg.profile.pre_samples + 2 * cfg.profile.direct_search_samples
    if needed >= cfg.period_samples:
        raise ValueError(
            f"max_range_m={cfg.profile.max_range_m} needs {needed:.0f} samples of lag "
            f"but one period is only {cfg.period_samples}. Lengthen chirp.period_s."
        )


class GainError(ValueError):
    """Raised when a requested chirp gain exceeds the configured safety ceiling."""


def resolve_gain(requested: float, max_gain: float, allow_loud: bool = False) -> float:
    """Return a safe playback gain.

    The configured ``max_gain`` cannot be exceeded unless ``allow_loud`` is set
    explicitly; digital full scale (1.0) can never be exceeded.
    """
    if requested < 0:
        raise GainError("Gain must be non-negative.")
    if requested > max_gain and not allow_loud:
        raise GainError(
            f"Requested gain {requested:.3f} exceeds safety.max_gain={max_gain:.3f}. "
            "Pass --allow-loud to override (keep speakers away from ears)."
        )
    return float(min(requested, FULL_SCALE))


_YAML_KEY_RE = r"^(\s*{key}\s*:\s*)([^#\n]*?)(\s*(#.*)?)$"


def update_yaml_value(path: str | Path, section: str, key: str, value: Any) -> None:
    """Rewrite one scalar in the YAML file in place, preserving comments/layout."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    in_section = False
    pattern = re.compile(_YAML_KEY_RE.format(key=re.escape(key)))
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(section)}\s*:", line):
            in_section = True
            continue
        if in_section and re.match(r"^\S", line):
            break  # left the section
        if in_section:
            m = pattern.match(line.rstrip("\r\n"))
            if m:
                newline = line[len(line.rstrip("\r\n")):]
                rendered = yaml.safe_dump(value, default_flow_style=True).strip()
                rendered = rendered.removesuffix("...").strip()
                lines[i] = f"{m.group(1)}{rendered}{m.group(3)}{newline}"
                p.write_text("".join(lines), encoding="utf-8")
                return
    raise KeyError(f"{section}.{key} not found in {p}")
