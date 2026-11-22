"""Live demo: waterfall plus model predictions (presence, distance, room).

    python -m echo.live_infer --name first_run
    python -m echo.live_infer --name sim --sim --sim-person wander --sim-room sim_lab
    python -m echo.live_infer --name sim --sim --sim-person wander --headless \
        --seconds 30 --save docs/live_infer_sim.png

Predictions are smoothed for display: presence uses hysteresis (on above 0.6,
off below 0.4), distance is a median of the last few frames, and the room is
an exponential average of the room probabilities.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from echo.config import REPO_ROOT, load_config
from echo.dataset import StreamingFeatures
from echo.live_waterfall import FrameFeed, Waterfall, run_reader
from echo.profile import EchoFrame
from echo.sonar import add_source_args, short_device_name, source_from_args
from echo.train import load_model, predict
from echo.viz import ACCENT, ACCENT2, FG

WARMUP_FRAMES = 20  # matches dataset.build_frameset(skip_frames=20)
GREEN, GREY = "#8fd694", "#6b7280"


@dataclass(frozen=True)
class Prediction:
    ready: bool
    presence_prob: float
    present: bool
    distance_m: float  # NaN when nobody
    room: str | None
    room_prob: float


class LivePredictor:
    """Frame-by-frame inference with display smoothing."""

    def __init__(self, checkpoint: Path, on: float = 0.6, off: float = 0.4,
                 median_frames: int = 7, room_alpha: float = 0.05) -> None:
        self.model, self.norm, self.rooms, _ = load_model(checkpoint)
        self.features = StreamingFeatures()
        self.on, self.off = on, off
        self.present = False
        self.p_smooth = 0.0
        self.dist_hist: deque[float] = deque(maxlen=median_frames)
        self.room_alpha = room_alpha
        self.room_p: np.ndarray | None = None
        self.n = 0

    def update(self, frames: list[EchoFrame]) -> list[Prediction]:
        if not frames:
            return []
        x = np.stack([self.features.update(f.profile, f.direct_amp) for f in frames])
        out = predict(self.model, self.norm(x))
        preds = []
        for i in range(len(frames)):
            self.n += 1
            p = float(out["presence"][i])
            self.p_smooth = 0.6 * self.p_smooth + 0.4 * p
            if self.present and self.p_smooth < self.off:
                self.present = False
                self.dist_hist.clear()
            elif not self.present and self.p_smooth > self.on:
                self.present = True
            if self.present:
                self.dist_hist.append(float(out["distance"][i]))
            room, room_prob = None, float("nan")
            if "room_proba" in out:
                rp = out["room_proba"][i]
                self.room_p = rp if self.room_p is None else (
                    (1 - self.room_alpha) * self.room_p + self.room_alpha * rp)
                k = int(np.argmax(self.room_p))
                room, room_prob = self.rooms[k], float(self.room_p[k])
            ready = self.n > WARMUP_FRAMES
            dist = float(np.median(self.dist_hist)) if self.present and self.dist_hist else float("nan")
            preds.append(Prediction(ready, self.p_smooth, self.present and ready, dist, room, room_prob))
        return preds


class InfoPanel:
    """Big, glanceable prediction readout in the bottom-right axes."""

    def __init__(self, ax: object) -> None:
        self.ax = ax
        kw = {"transform": ax.transAxes, "ha": "left", "va": "center"}  # type: ignore[attr-defined]
        self.state = ax.text(0.02, 0.84, "warming up", fontsize=22, fontweight="bold",  # type: ignore
                             color=GREY, **kw)
        self.dist = ax.text(0.02, 0.60, "", fontsize=15, color=FG, **kw)  # type: ignore
        self.room = ax.text(0.02, 0.40, "", fontsize=12, color=FG, **kw)  # type: ignore
        self.bar_bg = ax.barh([0.15], [1.0], height=0.10, left=0, color="#2e3440",  # type: ignore
                              transform=ax.transAxes)
        self.bar = ax.barh([0.15], [0.0], height=0.10, left=0, color=ACCENT,  # type: ignore
                           transform=ax.transAxes)[0]
        self.bar_label = ax.text(0.02, 0.03, "P(person)", fontsize=9, color=FG, **kw)  # type: ignore

    def show(self, p: Prediction | None) -> None:
        if p is None or not p.ready:
            return
        if p.present:
            self.state.set_text("PERSON")
            self.state.set_color(GREEN)
            self.dist.set_text(f"{p.distance_m:.2f} m away" if np.isfinite(p.distance_m) else "")
        else:
            self.state.set_text("EMPTY")
            self.state.set_color(GREY)
            self.dist.set_text("")
        self.room.set_text(f"room: {p.room}  ({100 * p.room_prob:.0f}%)" if p.room else "")
        self.bar.set_width(float(np.clip(p.presence_prob, 0, 1)))
        self.bar.set_color(GREEN if p.present else ACCENT2 if p.presence_prob > 0.3 else ACCENT)
        self.bar_label.set_text(f"P(person) {p.presence_prob:.2f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--name", default="run", help="run name under train.out_dir")
    ap.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    ap.add_argument("--history", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.headless and args.seconds is None:
        ap.error("--headless needs --seconds")
    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams["keymap.save"] = [k for k in plt.rcParams["keymap.save"] if k != "s"]
    torch.set_num_threads(2)
    cfg0 = load_config(args.config)
    ckpt = args.checkpoint or REPO_ROOT / cfg0.train.out_dir / args.name / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Train first: python -m echo.train --name {args.name}")
    predictor = LivePredictor(ckpt)
    cfg, source = source_from_args(args, realtime=not args.headless)
    max_frames = None if args.seconds is None else int(round(args.seconds / cfg.chirp.period_s))
    wf = Waterfall(cfg, history_s=args.history, info_panel=True,
                   title=f"echolocation live - {short_device_name(source.device_name)}")
    info = InfoPanel(wf.ax_info)
    feed = FrameFeed()

    def step() -> None:
        frames = feed.drain()
        wf.add(frames)
        preds = predictor.update(frames)
        wf.add_track([p.distance_m if p.present else np.nan for p in preds])
        wf.draw(source.xruns)
        info.show(preds[-1] if preds else None)

    if args.headless:
        run_reader(source, cfg, feed, max_frames=max_frames)
        if feed.error:
            raise feed.error
        step()
    else:
        reader = threading.Thread(target=run_reader, args=(source, cfg, feed, max_frames), daemon=True)
        reader.start()
        shot = args.save or Path("docs/live_infer.png")

        def on_key(event: object) -> None:
            if getattr(event, "key", None) == "s":
                shot.parent.mkdir(parents=True, exist_ok=True)
                wf.fig.savefig(shot, dpi=130)
                print(f"Screenshot saved to {shot}")

        wf.fig.canvas.mpl_connect("key_press_event", on_key)

        def update(_: int) -> tuple[object, ...]:
            step()
            if feed.error is not None:
                plt.close(wf.fig)
            return ()

        print("Press 's' to save a screenshot; close the window to stop.")
        anim = FuncAnimation(wf.fig, update, interval=int(cfg.chirp.period_s * 1000),
                             blit=False, cache_frame_data=False)
        plt.show()
        del anim
        feed.stop.set()
        reader.join(timeout=3)
        if feed.error is not None and not isinstance(feed.error, KeyboardInterrupt):
            raise feed.error
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        wf.fig.savefig(args.save, dpi=130)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
