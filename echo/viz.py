"""Plotting helpers shared by the live tools and offline reports.

CLI (side-by-side session waterfalls)::

    python -m echo.viz sessions data/sessions/a.npz data/sessions/b.npz --out docs/sessions.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from echo.profile import RangeGrid, ema_subtract

CMAP = "magma"
BG = "#0e1117"
FG = "#d8dee9"
ACCENT = "#5ec4ff"
ACCENT2 = "#ffb35e"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": "#3b4252",
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "text.color": FG,
        "grid.color": "#2e3440",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })


def to_db(x: np.ndarray, ref: np.ndarray | float = 1.0, floor_db: float = -100.0) -> np.ndarray:
    ratio = np.abs(np.asarray(x, dtype=np.float64)) / (np.asarray(ref, dtype=np.float64) + 1e-12)
    return np.maximum(20.0 * np.log10(ratio + 1e-12), floor_db)


def motion_db(profiles: np.ndarray, direct_amp: np.ndarray, alpha: float) -> np.ndarray:
    """Background-subtracted magnitude in dB relative to the direct path, (T, B)."""
    diff = ema_subtract(profiles, alpha)
    return to_db(diff, direct_amp[:, None])


def plot_waterfall(
    ax: Axes,
    image_db: np.ndarray,
    times: np.ndarray,
    grid: RangeGrid,
    title: str = "",
    vmin: float = -75.0,
    vmax: float = -30.0,
) -> object:
    """Range (y) versus time (x). ``image_db`` is (T, B)."""
    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) > 1 else t0 + 1.0
    im = ax.imshow(
        image_db.T,
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        extent=(t0, t1, 0.0, grid.n_bins * grid.bin_m),
        interpolation="nearest",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    if title:
        ax.set_title(title, loc="left")
    return im


def waterfall_figure(
    profiles: np.ndarray,
    times: np.ndarray,
    direct_amp: np.ndarray,
    grid: RangeGrid,
    alpha: float,
    title: str,
) -> Figure:
    apply_style()
    fig, (ax_w, ax_p) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True
    )
    img = motion_db(profiles, direct_amp, alpha)
    im = plot_waterfall(ax_w, img, times - times[0], grid, title)
    fig.colorbar(im, ax=ax_w, label="change vs background (dB re direct path)", pad=0.01)
    axis = grid.axis_m()
    ax_p.plot(to_db(profiles[-1], direct_amp[-1]), axis, color=ACCENT, lw=1.2, label="profile")
    ax_p.plot(img[-1], axis, color=ACCENT2, lw=1.0, label="minus background")
    ax_p.set_ylim(0, axis[-1])
    ax_p.set_xlim(-90, 0)
    ax_p.set_xlabel("dB re direct path")
    ax_p.set_title("latest frame", loc="left")
    ax_p.grid(True, alpha=0.4)
    ax_p.legend(loc="upper right", frameon=False, fontsize=8)
    return fig


def sessions_figure(paths: Sequence[Path], alpha: float) -> Figure:
    """Side-by-side waterfalls of recorded sessions (same colour scale)."""
    from echo.dataset import load_session

    apply_style()
    sessions = [load_session(p) for p in paths]
    fig, axes = plt.subplots(1, len(sessions), figsize=(6.2 * len(sessions), 4.8),
                             constrained_layout=True, squeeze=False)
    im = None
    for ax, s in zip(axes[0], sessions):
        img = motion_db(s.profiles, s.direct_amp, alpha)
        title = f"{s.meta.get('label', '?')}  |  room={s.meta.get('room', '?')}"
        im = plot_waterfall(ax, img, s.times - s.times[0], s.grid, title)
        if np.isfinite(s.distance).any():
            ax.plot(s.times - s.times[0], s.distance, color=ACCENT, lw=0.8, alpha=0.7,
                    label="label distance")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), label="change vs background (dB re direct path)", pad=0.01)
    return fig


def main(argv: list[str] | None = None) -> None:
    from echo.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions", help="side-by-side session waterfalls")
    s.add_argument("paths", nargs="+", type=Path)
    s.add_argument("--out", type=Path, default=Path("docs/sessions.png"))
    s.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fig = sessions_figure(args.paths, cfg.profile.background_alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
