"""Stage 2 (simulator): calibration, moving echo as a diagonal line, waterfall render."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from echo.calibrate import calibrate
from echo.config import load_config
from echo.live_waterfall import main as waterfall_main
from echo.profile import meters_to_delay, range_grid, stack_profiles
from echo.sonar import SimScene, open_source, stream_frames
from echo.viz import motion_db, waterfall_figure


def test_calibration_recovers_offset() -> None:
    cfg = load_config()
    res = calibrate(cfg, 0.5, sim=True, seconds=1.5, interactive=False)
    # The simulated speaker-mic direct path is 0.12 m one-way.
    expected = -float(meters_to_delay(cfg.sim.speaker_mic_m / 2, cfg.audio.sample_rate))
    assert abs(res.new_offset - expected) < 1.0, res.report()
    # After calibration the object is measured at the right distance.
    cal = dataclasses.replace(cfg, profile=dataclasses.replace(
        cfg.profile, calib_offset_samples=res.new_offset))
    res2 = calibrate(cal, 1.2, sim=True, seconds=1.5, interactive=False)
    assert abs(res2.error_m) < 0.005, res2.report()


def _moving_frames(seconds: float = 8.0):
    cfg = load_config()
    src = open_source(cfg, sim=True, scene=SimScene(person="linear:0.6:3.0:8", realtime=False))
    n = int(seconds / cfg.chirp.period_s)
    with src:
        frames = list(stream_frames(src, cfg, max_frames=n))
    truth = np.array([src.truth_at_capture(f.time_s) for f in frames])  # type: ignore[attr-defined]
    return cfg, frames, truth


def test_moving_echo_is_a_diagonal_line() -> None:
    cfg, frames, truth = _moving_frames()
    grid = range_grid(cfg)
    P = stack_profiles(frames)
    amps = np.array([f.direct_amp for f in frames])
    t = np.array([f.time_s for f in frames]) - frames[0].time_s
    img = motion_db(P, amps, cfg.profile.background_alpha)
    axis = grid.axis_m()
    near = axis > 0.4
    track = axis[near][np.argmax(img[:, near], axis=1)]
    # The EMA background starts from frame 0, which already contains the target,
    # leaving a fading "ghost" at the start position for a few seconds.
    sel = t > 3.0
    on_line = np.abs(track[sel] - truth[sel]) < 0.1
    assert on_line.mean() > 0.9
    slope = np.polyfit(t[sel][on_line], track[sel][on_line], 1)[0]
    assert abs(slope - 2.4 / 8.0) < 0.03  # 0.6 -> 3.0 m over 8 s: a straight diagonal
    fig = waterfall_figure(P, t, amps, grid, cfg.profile.background_alpha, "test")
    assert fig is not None


def test_live_waterfall_headless_saves_png(tmp_path: Path) -> None:
    out = tmp_path / "wf.png"
    waterfall_main(["--sim", "--sim-person", "wave", "--headless", "--seconds", "3",
                    "--save", str(out)])
    assert out.exists() and out.stat().st_size > 10_000
