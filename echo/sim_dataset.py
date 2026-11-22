"""Generate a synthetic multi-room, multi-day dataset with the simulator.

    python -m echo.sim_dataset --out data/sim_sessions --days 3

Each room/day gets: an empty session, a session of someone wandering in and
out of range (camera-style auto labels), and short sessions of a person
standing still at fixed distances (manual labels). Gain, system latency and
small room details vary per session, like real recordings would.

Useful for exercising training/evaluation end to end before real data exists.
Results on simulated data say nothing about real-world accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from echo.collect import collect_session
from echo.config import Config, load_config
from echo.simulate import PersonModel, SimSonar, make_room
from echo.sonar import parse_person

DEFAULT_ROOMS = ("sim_lab", "sim_dorm")


def generate(
    cfg: Config,
    out_dir: Path,
    rooms: tuple[str, ...] = DEFAULT_ROOMS,
    days: int = 3,
    scale: float = 1.0,
    seed: int = 0,
    progress: bool = True,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 5, 10, 0, 0)
    paths: list[Path] = []
    plan: list[tuple[str, str, float, bool]] = [
        ("empty", "none", 30, False),
        ("auto", "wander", 60, True),
        ("person_0.8m", "static:0.8", 15, False),
        ("person_1.5m", "static:1.5", 15, False),
        ("person_2.5m", "static:2.5", 15, False),
    ]
    k = 0
    for day in range(days):
        for room in rooms:
            for label, person, seconds, camera in plan:
                k += 1
                when = start + timedelta(days=day, minutes=k)
                sim = dataclasses.replace(
                    cfg.sim,
                    seed=int(rng.integers(1 << 30)),
                    latency_samples=int(rng.integers(500, 6000)),
                    noise_std=float(cfg.sim.noise_std * rng.uniform(0.8, 1.25)),
                )
                c = dataclasses.replace(cfg, sim=sim)
                gain = float(rng.uniform(0.08, 0.12))
                pm = PersonModel(strength=float(rng.uniform(0.009, 0.015)))
                src = SimSonar(c, room=make_room(room, variant=day + 1),
                               person=parse_person(person, seed=int(rng.integers(1 << 30))),
                               gain=gain, person_model=pm, seed=sim.seed)
                p = collect_session(c, src, room, label, seconds * scale, out_dir, camera=camera,
                                    progress=False, when=when,
                                    notes="simulated", extra_meta={"sim_person": person})
                paths.append(p)
                if progress:
                    print(f"  {p.name}")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/sim_sessions"))
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--rooms", nargs="+", default=list(DEFAULT_ROOMS))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every session length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"Writing simulated sessions to {args.out}")
    paths = generate(cfg, args.out, tuple(args.rooms), args.days, args.scale, args.seed)
    print(f"{len(paths)} sessions written.")


if __name__ == "__main__":
    main()
