"""One interface over the real full-duplex stream and the simulator.

Every live tool (diagnostics, waterfall, calibration, collection, inference)
takes ``--sim`` and gets the same stream of :class:`~echo.profile.EchoFrame`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from echo.config import Config, load_config, resolve_gain
from echo.profile import EchoFrame, EchoProcessor
from echo.simulate import (
    PersonModel,
    SimSonar,
    Trajectory,
    absent,
    hand_wave,
    linear_motion,
    make_room,
    random_walk,
    static_at,
)


class Source(Protocol):
    device_name: str
    t0: float
    xruns: int

    @property
    def gain(self) -> float: ...

    def set_gain(self, gain: float) -> None: ...

    def blocks(self) -> Iterator[np.ndarray]: ...

    def __enter__(self) -> Source: ...

    def __exit__(self, *exc: object) -> None: ...


@dataclass(frozen=True)
class SimScene:
    room: str = "sim_room"
    person: str = "none"  # see parse_person
    agc: bool = False
    realtime: bool = True
    seed: int | None = None


def short_device_name(name: str) -> str:
    """'in: Microphone Array (3- Intel(R) ...) | out: ...' -> 'in: Microphone Array | out: ...'."""
    prev = None
    while prev != name:  # strip innermost parentheses until none are left
        prev, name = name, re.sub(r"\s*\([^()]*\)", "", name)
    return name


def parse_person(spec: str, seed: int = 0) -> Trajectory:
    """Person trajectory from a short spec.

    ``none`` | ``static:D`` | ``wave[:CENTER]`` | ``linear:R0:R1:SECONDS`` |
    ``walk[:RMIN:RMAX]`` | ``wander[:RMIN:RMAX]`` (walk that sometimes leaves).
    """
    name, *vals = spec.split(":")
    nums = [float(v) for v in vals]
    rng = np.random.default_rng(seed)
    if name == "none":
        return absent()
    if name in ("static", "book"):
        return static_at(nums[0] if nums else 1.0)
    if name == "wave":
        return hand_wave(center_m=nums[0] if nums else 0.3)
    if name == "linear":
        r0, r1, dur = (nums + [0.5, 3.0, 10.0][len(nums):])[:3]
        return linear_motion(r0, r1, dur)
    if name in ("walk", "wander"):
        lo, hi = (nums + [0.5, 3.0][len(nums):])[:2]
        return random_walk(rng, lo, hi, leave_prob=0.25 if name == "wander" else 0.0)
    raise ValueError(f"Unknown person spec {spec!r}")


def open_source(
    cfg: Config,
    sim: bool,
    gain: float | None = None,
    allow_loud: bool = False,
    scene: SimScene = SimScene(),
    keep_raw: bool = False,
) -> Source:
    """Create (but do not start) a sonar source; use it as a context manager."""
    g = resolve_gain(cfg.chirp.gain if gain is None else gain, cfg.safety.max_gain, allow_loud)
    if sim:
        person = parse_person(scene.person, seed=scene.seed or 0)
        # A hand or a book is a single compact reflector; a person is several.
        if scene.person.startswith("wave"):
            model = PersonModel(strength=0.004, parts=((0.0, 1.0),), sway_m=0.0)
        elif scene.person.startswith("book"):
            model = PersonModel(strength=0.03, parts=((0.0, 1.0),), sway_m=0.0)
        else:
            model = PersonModel()
        return SimSonar(cfg, room=make_room(scene.room), person=person, gain=g,
                        realtime=scene.realtime, agc=scene.agc, person_model=model, seed=scene.seed)
    from echo.audio_io import DuplexSonar

    return DuplexSonar(cfg, gain=g, allow_loud=allow_loud, keep_raw=keep_raw)


LOST_FRAMES = 20  # 1 s at 20 chirps/s
LOST_FRACTION = 0.2


def stream_frames(
    source: Source,
    cfg: Config,
    max_frames: int | None = None,
    on_block: Callable[[np.ndarray], None] | None = None,
) -> Iterator[EchoFrame]:
    """Turn a running source into echo frames.

    If the stream reports an xrun (dropped or inserted samples) the period
    phase is no longer valid, so the processor is rebuilt and re-locks.

    If the chirp disappears (speaker muted, volume key, another app grabbing
    the device) every profile would be normalised by ~0 and look like a wall
    of motion. After ``LOST_FRAMES`` frames below ``LOST_FRACTION`` of the
    normal direct-path level the processor is rebuilt, so no frames are
    emitted until the chirp is back and locked again.

    ``on_block`` sees every raw captured block (e.g. for clipping checks).
    """
    proc: EchoProcessor | None = None
    consumed = 0
    xruns = source.xruns
    n = 0
    fs = cfg.audio.sample_rate
    ref_amps: list[float] = []  # direct-path level right after each lock
    low = 0
    lost = False
    for block in source.blocks():
        if on_block is not None:
            on_block(block)
        if proc is None or source.xruns != xruns:
            xruns = source.xruns
            proc = EchoProcessor(cfg, t0=source.t0 + consumed / fs)
        consumed += len(block)
        for f in proc.push(block):
            if lost:
                lost = False
                print("Chirp is back; re-locked.", file=sys.stderr)
            if len(ref_amps) < LOST_FRAMES:
                ref_amps.append(f.direct_amp)
            elif f.direct_amp < LOST_FRACTION * float(np.median(ref_amps)):
                low += 1
                if low >= LOST_FRAMES:
                    print("WARNING: the chirp has disappeared from the recording (speaker muted, "
                          "volume turned down, or the device taken by another app). Waiting for it "
                          "to come back...", file=sys.stderr)
                    proc = EchoProcessor(cfg, t0=source.t0 + consumed / fs)
                    ref_amps, low, lost = [], 0, True
                    break
            else:
                low = 0
            yield f
            n += 1
            if max_frames is not None and n >= max_frames:
                return


# --------------------------------------------------------------------------- #
# Shared CLI arguments
# --------------------------------------------------------------------------- #
def add_source_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of audio hardware")
    ap.add_argument("--gain", type=float, default=None, help="chirp gain (fraction of full scale)")
    ap.add_argument("--allow-loud", action="store_true", help="permit gain above safety.max_gain")
    ap.add_argument("--sim-room", default="sim_room", help="simulated room name (layout seed)")
    ap.add_argument("--sim-person", default="none",
                    help="simulated person: none | static:D | wave | linear:R0:R1:S | walk | wander")
    ap.add_argument("--sim-agc", action="store_true", help="simulate Windows AGC (linearity should fail)")
    ap.add_argument("--sim-seed", type=int, default=None)


def source_from_args(args: argparse.Namespace, realtime: bool = True,
                     keep_raw: bool = False) -> tuple[Config, Source]:
    cfg = load_config(args.config)
    scene = SimScene(room=args.sim_room, person=args.sim_person, agc=args.sim_agc,
                     realtime=realtime, seed=args.sim_seed)
    src = open_source(cfg, args.sim, gain=args.gain, allow_loud=args.allow_loud, scene=scene,
                      keep_raw=keep_raw)
    return cfg, src
