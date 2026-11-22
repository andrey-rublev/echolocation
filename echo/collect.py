"""Record a labelled session of echo profiles.

Manual labels (one label for the whole session)::

    python -m echo.collect --room dorm_desk --label empty --minutes 1
    python -m echo.collect --room dorm_desk --label person_1m --minutes 2
    python -m echo.collect --room kitchen --label walking --minutes 2 --notes "door open"

Camera auto-labels (per-frame presence and distance from the webcam)::

    python -m echo.collect --room dorm_desk --label auto --camera --minutes 3

Simulator (no hardware; useful for checking the pipeline)::

    python -m echo.collect --sim --sim-person wander --room sim_a --label auto --camera --minutes 1

Each session is written to ``collect.out_dir`` as ``<id>.npz`` (profiles,
timestamps, labels) plus ``<id>.json`` (metadata).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from echo import __version__
from echo.camera import CameraLabeler, align_labels, sim_camera_labels
from echo.config import REPO_ROOT, Config, load_config
from echo.dataset import parse_label, save_session
from echo.profile import EchoFrame, range_grid
from echo.simulate import SimSonar
from echo.sonar import Source, add_source_args, source_from_args, stream_frames


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in s.strip().lower())


def make_session_id(room: str, label: str, when: datetime) -> str:
    return f"{when:%Y%m%d_%H%M%S}_{_slug(room)}_{_slug(label)}"


def record_frames(cfg: Config, source: Source, seconds: float,
                  progress: bool = True) -> tuple[list[EchoFrame], bool]:
    """Record ``seconds`` of frames; returns (frames, interrupted)."""
    n = int(round(seconds / cfg.chirp.period_s))
    frames: list[EchoFrame] = []
    interrupted = False
    last_print = 0.0
    try:
        with source:
            for f in stream_frames(source, cfg, max_frames=n):
                frames.append(f)
                if progress and (len(frames) - last_print) >= 20:
                    last_print = len(frames)
                    pct = 100 * len(frames) / n
                    sys.stdout.write(f"\r  {pct:5.1f}%  {len(frames)}/{n} frames  "
                                     f"direct {f.direct_amp:.2f}  xruns {source.xruns}   ")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
    if progress:
        sys.stdout.write("\n")
    return frames, interrupted


def collect_session(
    cfg: Config,
    source: Source,
    room: str,
    label: str,
    seconds: float,
    out_dir: Path,
    camera: bool = False,
    presence: float | None = None,
    distance: float | None = None,
    notes: str = "",
    progress: bool = True,
    when: datetime | None = None,
    extra_meta: dict[str, Any] | None = None,
    save_wav: bool = False,
) -> Path:
    when = when or datetime.now()
    labeler: CameraLabeler | None = None
    is_sim = isinstance(source, SimSonar)
    if camera and not is_sim:
        labeler = CameraLabeler(cfg.camera)
        labeler.start()
    try:
        frames, interrupted = record_frames(cfg, source, seconds, progress)
    finally:
        if labeler is not None:
            labeler.stop()
    if len(frames) < 20:
        raise RuntimeError("Too few frames recorded; nothing saved.")
    times = np.array([f.time_s for f in frames])
    n = len(frames)

    if camera:
        if is_sim:
            assert isinstance(source, SimSonar)
            rng = np.random.default_rng(zlib.crc32(f"{room}|{label}|{when.isoformat()}".encode()))
            pres, dist = sim_camera_labels(times, source.truth_at_capture, rng)
        else:
            assert labeler is not None
            pres, dist = align_labels(times, labeler.samples, cfg.camera.max_label_age_s)
        label_source = "camera"
    else:
        p, d = parse_label(label)
        p = p if presence is None else presence
        d = d if distance is None else distance
        pres, dist = np.full(n, p, np.float32), np.full(n, d, np.float32)
        label_source = "manual"

    meta: dict[str, Any] = {
        "room": room,
        "label": label,
        "label_source": label_source,
        "device": getattr(source, "device_name", "unknown"),
        "gain": source.gain,
        "date": when.isoformat(timespec="seconds"),
        "day": when.strftime("%Y-%m-%d"),
        "notes": notes,
        "duration_s": float(times[-1] - times[0]) if n > 1 else 0.0,
        "xruns": source.xruns,
        "interrupted": interrupted,
        "simulated": is_sim,
        "camera_samples": len(labeler.samples) if labeler is not None else None,
        "version": __version__,
        "config": {"chirp": dataclasses.asdict(cfg.chirp), "profile": dataclasses.asdict(cfg.profile),
                   "audio": dataclasses.asdict(cfg.audio)},
    }
    if extra_meta:
        meta.update(extra_meta)
    sid = make_session_id(room, label, when)
    path = save_session(
        out_dir, sid,
        profiles=np.stack([f.profile for f in frames]),
        times=times,
        direct_amp=np.array([f.direct_amp for f in frames]),
        direct_lag=np.array([f.direct_lag for f in frames]),
        presence=pres, distance=dist, meta=meta, grid=range_grid(cfg),
    )
    if save_wav and not is_sim:
        from echo.audio_io import save_wavs

        save_wavs(out_dir / sid, cfg.audio.sample_rate, source.raw_capture,  # type: ignore[attr-defined]
                  source.raw_playback)  # type: ignore[attr-defined]
    return path


def summarize(path: Path) -> str:
    from echo.dataset import load_session

    s = load_session(path)
    pres = s.presence[np.isfinite(s.presence)]
    dist = s.distance[np.isfinite(s.distance)]
    lines = [f"Saved {path}", f"  frames: {len(s)}  ({s.meta.get('duration_s', 0):.1f} s)"]
    if len(pres):
        lines.append(f"  presence: {100 * pres.mean():.0f}% of labelled frames")
    if len(dist):
        lines.append(f"  distance: {dist.min():.2f}-{dist.max():.2f} m (median {np.median(dist):.2f})")
    if not len(pres):
        lines.append("  no presence labels (room-only session)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--room", required=True, help="room / placement name, e.g. dorm_desk")
    ap.add_argument("--label", required=True,
                    help="empty | person | walking | person_1m | person_1.5m | auto (with --camera)")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--camera", action="store_true", help="per-frame labels from the webcam")
    ap.add_argument("--presence", type=float, default=None, help="override presence label (0/1)")
    ap.add_argument("--distance", type=float, default=None, help="override distance label (m)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save-wav", action="store_true", help="also keep the raw capture as WAV")
    ap.add_argument("--countdown", type=float, default=None,
                    help="seconds to wait before recording (default 5 live, 0 sim)")
    args = ap.parse_args(argv)

    save_wav = args.save_wav or load_config(args.config).collect.save_wav
    cfg, source = source_from_args(args, realtime=False, keep_raw=save_wav and not args.sim)
    if not args.camera and math.isnan(parse_label(args.label)[0]) and args.presence is None:
        print(f"Note: label {args.label!r} carries no presence/distance; this session is useful "
              "for room identification only.")
    out_dir = args.out_dir or (REPO_ROOT / cfg.collect.out_dir)
    countdown = args.countdown if args.countdown is not None else (0.0 if args.sim else 5.0)
    print(f"Recording {args.minutes:g} min in room={args.room!r}, label={args.label!r} "
          f"({source.device_name}, gain {source.gain:.3f}). Ctrl+C stops early and saves.")
    if countdown > 0:
        for i in range(int(countdown), 0, -1):
            sys.stdout.write(f"\r  starting in {i}...  ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r  recording           \n")
    path = collect_session(cfg, source, args.room, args.label, args.minutes * 60, out_dir,
                           camera=args.camera, presence=args.presence, distance=args.distance,
                           notes=args.notes, save_wav=save_wav)
    print(summarize(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
