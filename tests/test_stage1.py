"""Stage 1 (simulator): duplex timing, linearity check and device selection."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from echo.audio_io import DeviceInfo, format_devices, resolve_device
from echo.chirp import PeriodicPlayer, make_period
from echo.config import GainError, load_config
from echo.diagnostics import run_linearity, run_timing
from echo.sonar import SimScene, open_source, parse_person


def test_timing_sim_captures_about_200_chirps() -> None:
    cfg = load_config()
    src = open_source(cfg, sim=True, scene=SimScene(realtime=False))
    res = run_timing(cfg, src, seconds=10.0)
    assert res.passed, res.report()
    assert 190 <= res.chirps_detected <= 200
    assert res.jitter_p2p < 0.5


def test_timing_flags_jitter() -> None:
    cfg = load_config()
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, jitter_samples=4))
    src = open_source(cfg, sim=True, scene=SimScene(realtime=False))
    res = run_timing(cfg, src, seconds=3.0)
    assert not res.passed and any("jitter" in p for p in res.problems)


def test_linearity_passes_on_linear_system() -> None:
    cfg = load_config()
    src = open_source(cfg, sim=True, scene=SimScene(realtime=False))
    res = run_linearity(cfg, src, step_seconds=1.5, steps=4)
    assert res.passed, res.report()
    assert all(abs(r - 2.0) < 0.1 for r in res.ratios)


def test_linearity_fails_with_agc() -> None:
    cfg = load_config()
    src = open_source(cfg, sim=True, scene=SimScene(realtime=False, agc=True))
    res = run_linearity(cfg, src, step_seconds=1.5, steps=4)
    assert not res.passed


def test_open_source_enforces_gain_ceiling() -> None:
    cfg = load_config()
    with pytest.raises(GainError):
        open_source(cfg, sim=True, gain=0.9)
    src = open_source(cfg, sim=True, gain=0.9, allow_loud=True)
    assert src.gain == 0.9


def test_player_gain_change_takes_effect() -> None:
    cfg = load_config()
    p = PeriodicPlayer(make_period(cfg.chirp, 48000), gain=0.1)
    a = p.fill(2400)
    p.gain = 0.05
    b = p.fill(2400)
    np.testing.assert_allclose(b, a / 2, rtol=1e-6)


def test_parse_person_specs() -> None:
    assert parse_person("none")(1.0) is None
    assert parse_person("static:2")(5.0) == 2.0
    lin = parse_person("linear:1:2:10")
    assert lin(0.0) == 1.0 and abs(lin(5.0) - 1.5) < 1e-9
    walk = parse_person("wander", seed=3)
    vals = [walk(t) for t in np.arange(0, 120, 0.5)]
    assert any(v is None for v in vals) and any(v is not None for v in vals)
    with pytest.raises(ValueError):
        parse_person("teleport")


DEVICES = [
    DeviceInfo(0, "Microphone Array (Realtek)", "MME", 2, 0, 44100),
    DeviceInfo(1, "Speakers (Realtek)", "MME", 0, 2, 44100),
    DeviceInfo(5, "Microphone Array (Realtek)", "Windows WASAPI", 2, 0, 48000),
    DeviceInfo(6, "Speakers (Realtek)", "Windows WASAPI", 0, 2, 48000),
]


def test_resolve_device_prefers_wasapi_by_name() -> None:
    assert resolve_device("microphone", "input", "Windows WASAPI", DEVICES) == 5
    assert resolve_device("speakers", "output", "Windows WASAPI", DEVICES) == 6
    assert resolve_device(1, "output", "Windows WASAPI", DEVICES) == 1
    with pytest.raises(ValueError):
        resolve_device(1, "input", "Windows WASAPI", DEVICES)
    with pytest.raises(ValueError):
        resolve_device("headset", "input", "Windows WASAPI", DEVICES)
    table = format_devices(DEVICES, "Windows WASAPI")
    assert table.splitlines()[1].lstrip().startswith("5*")
