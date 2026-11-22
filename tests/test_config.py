from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from echo.config import (
    DEFAULT_CONFIG_PATH,
    GainError,
    config_from_dict,
    load_config,
    resolve_gain,
    update_yaml_value,
)


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.audio.sample_rate == 48000
    assert cfg.period_samples == 2400
    assert cfg.chirp.gain <= cfg.safety.max_gain


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        config_from_dict({"chirp": {"bogus": 1}})


def test_range_must_fit_in_period() -> None:
    with pytest.raises(ValueError, match="period"):
        config_from_dict({"profile": {"max_range_m": 20.0}})


def test_gain_ceiling_enforced() -> None:
    assert resolve_gain(0.1, 0.3) == 0.1
    with pytest.raises(GainError):
        resolve_gain(0.5, 0.3)
    assert resolve_gain(0.5, 0.3, allow_loud=True) == 0.5
    assert resolve_gain(3.0, 0.3, allow_loud=True) == 1.0  # never above full scale
    with pytest.raises(GainError):
        resolve_gain(-0.1, 0.3)


def test_update_yaml_value_preserves_comments(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    shutil.copy(DEFAULT_CONFIG_PATH, p)
    before = p.read_text(encoding="utf-8")
    update_yaml_value(p, "profile", "calib_offset_samples", -16.75)
    after = p.read_text(encoding="utf-8")
    assert load_config(p).profile.calib_offset_samples == -16.75
    assert after.count("#") == before.count("#")
    with pytest.raises(KeyError):
        update_yaml_value(p, "profile", "nope", 1)
