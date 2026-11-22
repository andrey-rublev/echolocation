"""Stage 5: live inference (headless, simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from echo.config import load_config
from echo.live_infer import WARMUP_FRAMES, LivePredictor
from echo.live_infer import main as infer_main
from echo.sonar import SimScene, open_source, stream_frames


def test_live_predictor_streams(tiny_run: tuple[Path, Path]) -> None:
    _, run = tiny_run
    cfg = load_config()
    pred = LivePredictor(run / "best.pt")
    src = open_source(cfg, sim=True, scene=SimScene(person="static:1.5", realtime=False,
                                                    room="sim_lab"))
    with src:
        frames = list(stream_frames(src, cfg, max_frames=60))
    out = []
    for i in range(0, len(frames), 7):  # irregular batches, like UI ticks
        out += pred.update(frames[i : i + 7])
    assert len(out) == 60
    assert not out[WARMUP_FRAMES - 1].ready and out[-1].ready
    assert all(0.0 <= p.presence_prob <= 1.0 for p in out)
    assert all(np.isnan(p.distance_m) or 0 < p.distance_m < 4.2 for p in out)
    assert out[-1].room in ("sim_dorm", "sim_lab")


def test_live_infer_headless_png(tiny_run: tuple[Path, Path], tmp_path: Path) -> None:
    _, run = tiny_run
    out = tmp_path / "infer.png"
    infer_main(["--checkpoint", str(run / "best.pt"), "--sim", "--sim-person", "walk",
                "--headless", "--seconds", "4", "--save", str(out)])
    assert out.exists() and out.stat().st_size > 10_000
