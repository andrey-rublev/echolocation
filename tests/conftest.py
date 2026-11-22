from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # tests never open windows

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny simulated dataset and a 2-epoch training run, shared by tests.

    Returns (data_dir, run_dir). Kept small so the suite stays light on CPU.
    """
    from echo.config import load_config
    from echo.sim_dataset import generate
    from echo.train import train

    cfg = load_config()
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, num_threads=2, batch_size=64))
    base = tmp_path_factory.mktemp("tiny")
    data = base / "sessions"
    generate(cfg, data, days=1, scale=0.15, progress=False)
    run = base / "run"
    train(cfg, data, run, epochs=2, cooldown_s=0.0, verbose=False)
    return data, run
