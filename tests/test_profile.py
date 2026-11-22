from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from echo.config import Config, load_config
from echo.profile import (
    EchoProcessor,
    EmaBackground,
    delay_to_meters,
    ema_subtract,
    meters_to_delay,
    range_grid,
    stack_profiles,
)
from echo.simulate import Room, SimSonar, static_at


def sim_cfg(**profile_kw: object) -> Config:
    """Config whose calibration offset cancels the simulated speaker-mic path."""
    cfg = load_config()
    direct = float(meters_to_delay(cfg.sim.speaker_mic_m / 2, cfg.audio.sample_rate))
    prof = dataclasses.replace(cfg.profile, calib_offset_samples=-direct, **profile_kw)
    return dataclasses.replace(cfg, profile=prof)


def test_delay_meter_roundtrip() -> None:
    assert np.isclose(delay_to_meters(meters_to_delay(1.7, 48000), 48000), 1.7)
    # 1 m round trip at 48 kHz is ~280 samples.
    assert np.isclose(meters_to_delay(1.0, 48000), 279.88, atol=0.01)


def test_range_grid() -> None:
    cfg = load_config()
    g = range_grid(cfg)
    assert g.n_range_samples % g.decimate == 0
    assert g.axis_m()[-1] <= cfg.profile.max_range_m + g.bin_m
    assert g.n_bins == len(g.axis_m())


@pytest.mark.parametrize("latency", [137, 2399, 3517, 5000])
@pytest.mark.parametrize("align", ["direct", "fixed"])
def test_stream_locks_and_finds_target(latency: int, align: str) -> None:
    cfg = sim_cfg(align=align)
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, latency_samples=latency))
    room = Room("bare", reflectors=())
    sonar = SimSonar(cfg, room=room, person=static_at(1.5))
    proc = EchoProcessor(cfg)
    frames = []
    for block in sonar.blocks(40):
        # Irregular block sizes, like a real callback.
        for part in np.array_split(block, 3):
            frames.extend(proc.push(part))
    assert proc.lock is not None and proc.lock.spread <= 1
    assert len(frames) > 20
    # Consecutive period indices, one frame per period.
    idx = [f.index for f in frames]
    assert idx == list(range(idx[0], idx[0] + len(idx)))
    prof = stack_profiles(frames[5:]).mean(axis=0)
    grid = range_grid(cfg)
    far = grid.axis_m() > 0.5  # skip the direct-path skirt
    peak_m = grid.axis_m()[far][np.argmax(prof[far])]
    assert abs(peak_m - 1.5) < 0.03


def test_direct_timing_is_stable_without_jitter() -> None:
    cfg = sim_cfg()
    proc = EchoProcessor(cfg)
    frames = []
    for b in SimSonar(cfg).blocks(60):
        frames.extend(proc.push(b))
    lags = np.array([f.direct_lag for f in frames])
    assert np.ptp(lags) < 0.2


def test_direct_alignment_absorbs_jitter() -> None:
    cfg = sim_cfg(align="direct")
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, jitter_samples=3))
    sonar = SimSonar(cfg, room=Room("bare", ()), person=static_at(2.0))
    proc = EchoProcessor(cfg)
    frames = []
    for b in sonar.blocks(80):
        frames.extend(proc.push(b))
    lags = np.array([f.direct_lag for f in frames])
    assert np.ptp(lags) >= 2  # jitter is visible in the direct-path timing ...
    grid = range_grid(cfg)
    peaks = []
    for f in frames:
        far = grid.axis_m() > 0.5
        peaks.append(grid.axis_m()[far][np.argmax(f.profile[far])])
    assert np.std(peaks) < 0.01  # ... but the aligned target range is steady


def test_ema_background_matches_vectorised() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 7))
    bg = EmaBackground(0.1)
    streamed = np.stack([bg.update(r) for r in x])
    np.testing.assert_allclose(streamed, ema_subtract(x, 0.1), atol=1e-12)
    assert np.allclose(streamed[0], 0)


def test_background_removes_static_and_keeps_motion() -> None:
    cfg = sim_cfg()
    room = Room("r", reflectors=((1.0, 0.01),))
    sonar = SimSonar(cfg, room=room, person=lambda t: None if t < 2.0 else 1.5)
    proc = EchoProcessor(cfg)
    frames = []
    for b in sonar.blocks(70):
        frames.extend(proc.push(b))
    P = stack_profiles(frames)
    times = np.array([f.time_s for f in frames]) - frames[0].time_s
    diff = ema_subtract(P, cfg.profile.background_alpha)
    grid = range_grid(cfg)
    wall, person = grid.bin_of(1.0), grid.bin_of(1.5)
    before = times < 1.5
    onset = np.argmax(times >= 2.0 + 0.1)
    assert np.abs(diff[before][5:, wall]).max() < 0.1 * P[before][:, wall].mean()
    assert diff[onset, person - 2 : person + 3].max() > 3 * np.abs(diff[before][5:, person]).max()
