"""Real-time scrolling range-vs-time waterfall plus the current profile (~20 fps).

    python -m echo.live_waterfall                     # live audio
    python -m echo.live_waterfall --sim --sim-person wave
    python -m echo.live_waterfall --sim --sim-person linear:0.5:3:8 --headless \
        --seconds 16 --save docs/waterfall_sim.png

A background thread reads the audio stream and turns it into profiles; the
matplotlib timer only redraws, so a slow redraw never drops audio.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from echo.config import Config
from echo.profile import EchoFrame, EmaBackground, range_grid
from echo.sonar import Source, add_source_args, source_from_args, stream_frames
from echo.viz import ACCENT, ACCENT2, CMAP, apply_style, to_db


@dataclass
class FrameFeed:
    """Thread-safe hand-off of frames from the audio thread to the UI."""

    maxlen: int = 4000
    _q: deque[EchoFrame] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None

    def put(self, f: EchoFrame) -> None:
        with self._lock:
            self._q.append(f)
            while len(self._q) > self.maxlen:
                self._q.popleft()

    def drain(self) -> list[EchoFrame]:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out


def run_reader(source: Source, cfg: Config, feed: FrameFeed, max_frames: int | None = None) -> None:
    try:
        with source:
            for f in stream_frames(source, cfg, max_frames=max_frames):
                feed.put(f)
                if feed.stop.is_set():
                    break
    except BaseException as exc:  # surfaced in the UI thread
        feed.error = exc
    finally:
        feed.stop.set()


class Waterfall:
    """Holds the scrolling image and draws it into a matplotlib figure."""

    def __init__(self, cfg: Config, history_s: float = 10.0, title: str = "") -> None:
        import matplotlib.pyplot as plt

        self.cfg = cfg
        self.grid = range_grid(cfg)
        self.n_cols = int(round(history_s / cfg.chirp.period_s))
        self.bg = EmaBackground(cfg.profile.background_alpha)
        self.img = np.full((self.grid.n_bins, self.n_cols), -120.0)
        self.frames_seen = 0
        self.last: EchoFrame | None = None
        self.last_diff = np.zeros(self.grid.n_bins)
        self.first_t: float | None = None
        self.track: deque[float] = deque(maxlen=self.n_cols)

        apply_style()
        self.fig, (self.ax_w, self.ax_p) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
        )
        top = self.grid.n_bins * self.grid.bin_m
        self.im = self.ax_w.imshow(self.img, origin="lower", aspect="auto", cmap=CMAP,
                                   vmin=-75, vmax=-30, extent=(-history_s, 0, 0, top),
                                   interpolation="nearest")
        self.ax_w.set_xlabel("time (s, now = 0)")
        self.ax_w.set_ylabel("range (m)")
        self.ax_w.set_title(title or "echo waterfall (change vs background)", loc="left")
        self.fig.colorbar(self.im, ax=self.ax_w, label="dB re direct path", pad=0.01)
        axis = self.grid.axis_m()
        (self.l_raw,) = self.ax_p.plot(np.full_like(axis, -120), axis, color=ACCENT, lw=1.2,
                                       label="profile")
        (self.l_diff,) = self.ax_p.plot(np.full_like(axis, -120), axis, color=ACCENT2, lw=1.0,
                                        label="minus background")
        self.ax_p.set_xlim(-90, 0)
        self.ax_p.set_ylim(0, top)
        self.ax_p.set_xlabel("dB re direct path")
        self.ax_p.set_title("current profile", loc="left")
        self.ax_p.grid(True, alpha=0.4)
        self.ax_p.legend(loc="upper right", frameon=False, fontsize=8)
        self.status = self.ax_w.text(0.01, 0.97, "", transform=self.ax_w.transAxes, va="top",
                                     fontsize=9, color="white",
                                     bbox={"facecolor": "black", "alpha": 0.5, "lw": 0})

    def add(self, frames: list[EchoFrame]) -> None:
        if not frames:
            return
        cols = []
        for f in frames:
            diff = self.bg.update(f.profile)
            cols.append(to_db(diff, f.direct_amp))
            self.last, self.last_diff = f, diff
            self.first_t = f.time_s if self.first_t is None else self.first_t
            self.frames_seen += 1
        new = np.stack(cols, axis=1)[:, -self.n_cols:]
        k = new.shape[1]
        self.img = np.concatenate([self.img[:, k:], new], axis=1)

    def draw(self, xruns: int = 0) -> None:
        self.im.set_data(self.img)
        if self.last is not None:
            f = self.last
            self.l_raw.set_xdata(to_db(f.profile, f.direct_amp))
            self.l_diff.set_xdata(to_db(self.last_diff, f.direct_amp))
            elapsed = f.time_s - (self.first_t or f.time_s)
            fps = self.frames_seen / elapsed if elapsed > 0 else 0.0
            self.status.set_text(f"{fps:4.1f} fps  |  direct {f.direct_amp:.2f}  |  xruns {xruns}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--history", type=float, default=10.0, help="seconds shown")
    ap.add_argument("--headless", action="store_true", help="no window; render and save at the end")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--save", type=Path, default=None, help="save a PNG screenshot here on exit")
    args = ap.parse_args(argv)
    if args.headless and args.seconds is None:
        ap.error("--headless needs --seconds")

    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # 's' saves our screenshot; stop matplotlib from also opening its save dialog.
    plt.rcParams["keymap.save"] = [k for k in plt.rcParams["keymap.save"] if k != "s"]
    cfg, source = source_from_args(args, realtime=not args.headless)
    max_frames = None if args.seconds is None else int(round(args.seconds / cfg.chirp.period_s))
    wf = Waterfall(cfg, history_s=args.history,
                   title=f"echo waterfall - {source.device_name}")
    feed = FrameFeed()

    if args.headless:
        run_reader(source, cfg, feed, max_frames=max_frames)
        if feed.error:
            raise feed.error
        wf.add(feed.drain())
        wf.draw(source.xruns)
    else:
        reader = threading.Thread(target=run_reader, args=(source, cfg, feed, max_frames), daemon=True)
        reader.start()

        def update(_: int) -> tuple[object, ...]:
            wf.add(feed.drain())
            wf.draw(source.xruns)
            if feed.error is not None:
                plt.close(wf.fig)
            return wf.im, wf.l_raw, wf.l_diff, wf.status

        shot = args.save or Path("docs/waterfall_live.png")

        def on_key(event: object) -> None:
            if getattr(event, "key", None) == "s":
                shot.parent.mkdir(parents=True, exist_ok=True)
                wf.fig.savefig(shot, dpi=130)
                print(f"Screenshot saved to {shot}")

        wf.fig.canvas.mpl_connect("key_press_event", on_key)
        print("Press 's' in the window to save a screenshot, close the window to stop.")
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
