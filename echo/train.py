"""Train EchoNet on recorded sessions (split by session/day, never by row).

    python -m echo.train --data data/sessions --name first_run
    python -m echo.train --data data/sim_sessions --name sim --epochs 15
    python -m echo.train --name first_run --resume      # continue after a crash

CPU only, ``train.num_threads`` threads (default 4), a checkpoint after every
epoch (``last.pt``) plus ``best.pt`` on validation improvement, and a short
pause between epochs to keep the laptop cool.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from echo.config import REPO_ROOT, Config, TrainConfig, load_config
from echo.dataset import (
    FEATURE_CHANNELS,
    FrameSet,
    Normalizer,
    Session,
    Split,
    augment,
    build_frameset,
    load_sessions,
    split_sessions,
)
from echo.model import EchoNet, EchoNetConfig, count_parameters

MAX_PARAMS = 100_000
DIST_BETA = 0.1  # SmoothL1 transition (m)
W_PRESENCE, W_DISTANCE, W_RANGE_CE, W_ROOM = 1.0, 2.0, 0.2, 0.5


@dataclass
class Prepared:
    sessions: list[Session]
    split: Split
    rooms: tuple[str, ...]
    train: FrameSet
    val: FrameSet | None
    test: FrameSet
    norm: Normalizer
    bin_m: float
    n_bins: int


def load_data(data_dir: Path) -> list[Session]:
    sessions = load_sessions(data_dir)
    if not sessions:
        raise SystemExit(f"No sessions in {data_dir}. Record some with `python -m echo.collect`.")
    bins = {s.profiles.shape[1] for s in sessions}
    if len(bins) > 1:
        common = max(bins, key=lambda b: sum(s.profiles.shape[1] == b for s in sessions))
        dropped = [s.id for s in sessions if s.profiles.shape[1] != common]
        print(f"Skipping {len(dropped)} session(s) recorded with a different range grid: {dropped}")
        sessions = [s for s in sessions if s.profiles.shape[1] == common]
    return sessions


def prepare(sessions: list[Session], split: Split) -> Prepared:
    by_id = {s.id: s for s in sessions}
    missing = [i for i in split.train + split.val + split.test if i not in by_id]
    if missing:
        raise SystemExit(f"Split refers to sessions that are not on disk: {missing[:5]}")
    pick = lambda ids: [by_id[i] for i in ids]  # noqa: E731
    rooms = tuple(sorted({s.room for s in pick(split.train)}))
    train = build_frameset(pick(split.train), rooms)
    val = build_frameset(pick(split.val), rooms) if split.val else None
    test = build_frameset(pick(split.test), rooms)
    norm = Normalizer.fit(train.x)
    g = sessions[0].grid
    return Prepared(sessions, split, rooms, train, val, test, norm, g.bin_m, g.n_bins)


def make_model(prep: Prepared, seed: int = 0) -> EchoNet:
    torch.manual_seed(seed)
    m = EchoNet(EchoNetConfig(n_bins=prep.n_bins, bin_m=prep.bin_m,
                              in_channels=len(FEATURE_CHANNELS), n_rooms=len(prep.rooms)))
    n = count_parameters(m)
    if n > MAX_PARAMS:
        raise RuntimeError(f"Model has {n} parameters (> {MAX_PARAMS}).")
    return m


def losses(out: dict[str, torch.Tensor], presence: torch.Tensor, distance: torch.Tensor,
           room: torch.Tensor, pos_weight: torch.Tensor, bin_m: float) -> dict[str, torch.Tensor]:
    """Masked multi-task loss: each head only learns from frames labelled for it."""
    zero = out["presence"].sum() * 0.0
    res: dict[str, torch.Tensor] = {}
    pm = torch.isfinite(presence)
    res["presence"] = (nn.functional.binary_cross_entropy_with_logits(
        out["presence"][pm], presence[pm], pos_weight=pos_weight) if pm.any() else zero)
    dm = torch.isfinite(distance) & (presence == 1)
    if dm.any():
        res["distance"] = nn.functional.smooth_l1_loss(out["distance"][dm], distance[dm], beta=DIST_BETA)
        n4 = out["range_logits"].shape[1]
        target = torch.clamp(((distance[dm] / bin_m - 2.0) / 4.0).round().long(), 0, n4 - 1)
        res["range_ce"] = nn.functional.cross_entropy(out["range_logits"][dm], target)
    else:
        res["distance"] = res["range_ce"] = zero
    rm = room >= 0
    res["room"] = (nn.functional.cross_entropy(out["room"][rm], room[rm])
                   if "room" in out and rm.any() else zero)
    res["total"] = (W_PRESENCE * res["presence"] + W_DISTANCE * res["distance"]
                    + W_RANGE_CE * res["range_ce"] + W_ROOM * res["room"])
    return res


def _tensors(fs: FrameSet, idx: np.ndarray, x: np.ndarray) -> tuple[torch.Tensor, ...]:
    return (torch.from_numpy(x), torch.from_numpy(fs.presence[idx]),
            torch.from_numpy(fs.distance[idx]), torch.from_numpy(fs.room[idx]))


@torch.no_grad()
def predict(model: EchoNet, x: np.ndarray, batch: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(x), batch):
        o = model(torch.from_numpy(x[i : i + batch]))
        for k, v in o.items():
            outs.setdefault(k, []).append(v.numpy())
    res = {k: np.concatenate(v) for k, v in outs.items()}
    res["presence"] = 1.0 / (1.0 + np.exp(-res["presence"]))
    if "room" in res:
        res["room_proba"] = np.exp(res["room"] - res["room"].max(1, keepdims=True))
        res["room_proba"] /= res["room_proba"].sum(1, keepdims=True)
        res["room"] = np.argmax(res["room"], axis=1)
    return res


@torch.no_grad()
def evaluate_loss(model: EchoNet, fs: FrameSet, norm: Normalizer, pos_weight: torch.Tensor,
                  bin_m: float, batch: int = 1024) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(fs), batch):
        idx = np.arange(i, min(i + batch, len(fs)))
        x, p, d, r = _tensors(fs, idx, norm(fs.x[idx]))
        total += float(losses(model(x), p, d, r, pos_weight, bin_m)["total"]) * len(idx)
        n += len(idx)
    return total / max(n, 1)


def save_checkpoint(path: Path, model: EchoNet, opt: torch.optim.Optimizer,
                    sched: torch.optim.lr_scheduler.LRScheduler, epoch: int,
                    best: float, prep: Prepared, history: list[dict[str, float]]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": epoch,
        "best_val": best,
        "model_config": asdict(model.cfg),
        "normalizer": prep.norm.to_json(),
        "rooms": list(prep.rooms),
        "history": history,
    }, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the checkpoint


def load_model(path: Path) -> tuple[EchoNet, Normalizer, list[str], dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = EchoNet(EchoNetConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Normalizer.from_json(ck["normalizer"]), list(ck["rooms"]), ck


def train(cfg: Config, data_dir: Path, run_dir: Path, resume: bool = False,
          epochs: int | None = None, cooldown_s: float = 0.5, verbose: bool = True) -> Path:
    tc: TrainConfig = cfg.train
    torch.set_num_threads(tc.num_threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_data(data_dir)
    split_path = run_dir / "split.json"
    if resume and split_path.exists():
        split = Split.from_json(json.loads(split_path.read_text()))
    else:
        split = split_sessions(sessions, tc.test_fraction, tc.val_fraction, tc.seed)
        split_path.write_text(json.dumps({**split.to_json(), "data_dir": str(data_dir)}, indent=2))
    prep = prepare(sessions, split)
    model = make_model(prep, tc.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_epochs = epochs or tc.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    ok = np.isfinite(prep.train.presence)
    pos = float(prep.train.presence[ok].mean()) if ok.any() else 0.5
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-3), dtype=torch.float32)

    start_epoch, best, history = 0, math.inf, []
    last = run_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best_val"], ck["history"]
        if verbose:
            print(f"Resuming at epoch {start_epoch}")

    if verbose:
        print(f"Sessions: train {len(split.train)}, val {len(split.val)}, test {len(split.test)} "
              f"(grouped by {split.group_by}); frames: train {len(prep.train)}, "
              f"test {len(prep.test)}; rooms: {list(prep.rooms)}")
        print(f"EchoNet: {count_parameters(model):,} parameters, {tc.num_threads} threads")
    rng = np.random.default_rng(tc.seed + start_epoch)
    stale = 0
    for epoch in range(start_epoch, n_epochs):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(prep.train))
        run_loss, seen = 0.0, 0
        for i in range(0, len(order), tc.batch_size):
            idx = np.sort(order[i : i + tc.batch_size])
            x = augment(prep.norm(prep.train.x[idx]), tc.augment, rng, prep.norm)
            xt, p, d, r = _tensors(prep.train, idx, x)
            loss = losses(model(xt), p, d, r, pos_weight, prep.bin_m)["total"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        sched.step()
        train_loss = run_loss / max(seen, 1)
        val_loss = (evaluate_loss(model, prep.val, prep.norm, pos_weight, prep.bin_m)
                    if prep.val is not None else train_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": time.perf_counter() - t0})
        improved = val_loss < best
        if improved:
            best, stale = val_loss, 0
            save_checkpoint(run_dir / "best.pt", model, opt, sched, epoch, best, prep, history)
        else:
            stale += 1
        save_checkpoint(last, model, opt, sched, epoch, best, prep, history)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{n_epochs}  train {train_loss:.4f}  val {val_loss:.4f}"
                  f"{'  *' if improved else ''}  ({history[-1]['seconds']:.1f} s)")
        if stale >= tc.patience:
            if verbose:
                print(f"Early stop: no validation improvement for {tc.patience} epochs.")
            break
        if cooldown_s > 0:
            time.sleep(cooldown_s)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return run_dir / "best.pt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", type=Path, default=None, help="session directory")
    ap.add_argument("--name", default="run", help="run name under train.out_dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cooldown", type=float, default=0.5, help="pause between epochs (s)")
    ap.add_argument("--threads", type=int, default=None, help="override train.num_threads")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.threads is not None:
        cfg = replace(cfg, train=replace(cfg.train, num_threads=args.threads))
    data = args.data or REPO_ROOT / cfg.collect.out_dir
    run_dir = REPO_ROOT / cfg.train.out_dir / args.name
    best = train(cfg, data, run_dir, resume=args.resume, epochs=args.epochs, cooldown_s=args.cooldown)
    print(f"Best checkpoint: {best}\nNext: python -m echo.evaluate --name {args.name}")


if __name__ == "__main__":
    main()
