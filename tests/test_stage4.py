"""Stage 4: models, losses, baselines, and a tiny end-to-end train/evaluate run."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from echo.baselines import EnergyDetector, LinearBaseline, pooled_features
from echo.config import load_config
from echo.dataset import FrameSet, Normalizer, augment
from echo.evaluate import evaluate_run
from echo.metrics import auroc, best_threshold, binary_metrics, class_metrics, regression_metrics
from echo.model import EchoNet, EchoNetConfig, count_parameters
from echo.profile import range_grid
from echo.train import load_model, losses, train


def _net(n_rooms: int = 3) -> tuple[EchoNet, int]:
    g = range_grid(load_config())
    return EchoNet(EchoNetConfig(n_bins=g.n_bins, bin_m=g.bin_m, n_rooms=n_rooms)), g.n_bins


def test_model_is_small_and_shapes_are_right() -> None:
    net, b = _net()
    assert count_parameters(net) < 100_000
    out = net(torch.randn(5, 3, b))
    assert out["presence"].shape == (5,)
    assert out["distance"].shape == (5,)
    assert out["room"].shape == (5, 3)
    assert torch.all(out["distance"] > 0) and torch.all(out["distance"] < 4.1)
    net1, _ = _net(n_rooms=1)
    assert "room" not in net1(torch.randn(2, 3, b))


def test_losses_are_masked() -> None:
    net, b = _net()
    out = net(torch.randn(4, 3, b))
    nan = float("nan")
    pres = torch.tensor([nan, nan, nan, nan])
    dist = torch.tensor([nan, nan, nan, nan])
    room = torch.tensor([-1, -1, -1, -1])
    res = losses(out, pres, dist, room, torch.tensor(1.0), 0.0143)
    assert res["total"].item() == 0.0
    pres = torch.tensor([1.0, 0.0, 1.0, nan])
    dist = torch.tensor([1.5, nan, 2.0, nan])
    room = torch.tensor([0, 2, -1, 1])
    res = losses(out, pres, dist, room, torch.tensor(1.0), 0.0143)
    assert all(torch.isfinite(v) and v > 0 for v in res.values())
    res["total"].backward()


def test_metrics() -> None:
    y = np.array([0, 0, 1, 1, np.nan])
    s = np.array([0.1, 0.6, 0.4, 0.9, 0.5])
    m = binary_metrics(y, s)
    assert m.n == 4 and m.accuracy == 0.5 and m.confusion == [[1, 1], [1, 1]]
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    r = regression_metrics(np.array([1.0, 2.0, np.nan]), np.array([1.1, 2.5, 3.0]))
    assert r.n == 2 and r.mae == pytest.approx(0.3) and r.within_25cm == 0.5
    c = class_metrics(np.array([0, 1, 1, -1]), np.array([0, 1, 0, 1]), ["a", "b"])
    assert c.n == 3 and c.confusion == [[1, 0], [1, 1]]
    t = best_threshold(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def _toy_frameset(n: int = 400, b: int = 40, seed: int = 0) -> FrameSet:
    rng = np.random.default_rng(seed)
    # Presence comes in runs of 25 frames, like a person staying for a while.
    pres = np.repeat(rng.random(n // 25) > 0.5, 25).astype(np.float32)
    dist = np.where(pres == 1, rng.uniform(0.5, 3.0, n), np.nan).astype(np.float32)
    x = rng.normal(-3, 0.1, (n, 3, b)).astype(np.float32)
    for i in np.flatnonzero(pres == 1):
        k = int(dist[i] / 0.1)
        x[i, :, k] += 1.0
    return FrameSet(x, pres, dist, rng.integers(0, 2, n), np.repeat(np.arange(4), n // 4),
                    ("a", "b", "c", "d"), ("r0", "r1"))


def test_baselines_learn_toy_problem() -> None:
    fs = _toy_frameset()
    e = EnergyDetector(bin_m=0.1).fit(fs)
    assert binary_metrics(fs.presence, e.predict_proba(fs)).balanced_accuracy > 0.9
    lin = LinearBaseline(pool=1).fit(fs)
    out = lin.predict(fs)
    assert binary_metrics(fs.presence, out["presence"]).accuracy > 0.95
    ok = fs.presence == 1
    assert regression_metrics(fs.distance[ok], out["distance"][ok]).mae < 0.3
    assert pooled_features(fs.x, 4).shape == (400, 30)


def test_augment_preserves_shape_and_is_bounded() -> None:
    fs = _toy_frameset()
    norm = Normalizer.fit(fs.x)
    x = norm(fs.x[:32])
    cfg = load_config().train.augment
    y = augment(x, cfg, np.random.default_rng(0), norm)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert 0 < np.abs(y - x).mean() < 1.0


@pytest.mark.slow
def test_train_and_evaluate_end_to_end(tiny_run: tuple[Path, Path], tmp_path: Path) -> None:
    data, run0 = tiny_run
    assert len(list(data.glob("*.npz"))) == 10
    run = tmp_path / "run"
    shutil.copytree(run0, run)
    best = run / "best.pt"
    assert best.exists() and (run / "last.pt").exists() and (run / "split.json").exists()
    model, norm, rooms, ck = load_model(best)
    assert count_parameters(model) < 100_000 and rooms == ["sim_dorm", "sim_lab"]
    # Resume continues from the last epoch rather than starting over.
    cfg = load_config()
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, num_threads=2, batch_size=64))
    train(cfg, data, run, resume=True, epochs=3, cooldown_s=0.0, verbose=False)
    hist = json.loads((run / "history.json").read_text())
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    res = evaluate_run(run)
    assert (run / "results.md").exists() and (run / "metrics.json").exists()
    assert "cnn" in res.presence and res.verdict
    split = json.loads((run / "split.json").read_text())
    assert not set(split["test"]) & set(split["train"])
