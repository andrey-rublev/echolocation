"""Evaluate a trained run on its held-out test sessions, against the baselines.

    python -m echo.evaluate --name first_run

Writes to ``runs/<name>/``: ``metrics.json``, ``results.md`` (the table for the
README), ``confusion.png``, ``distance_error.png`` and ``timeline.png``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo.baselines import EnergyDetector, LinearBaseline  # noqa: E402
from echo.config import REPO_ROOT, load_config  # noqa: E402
from echo.dataset import FrameSet, Split  # noqa: E402
from echo.metrics import binary_metrics, class_metrics, regression_metrics  # noqa: E402
from echo.train import load_data, load_model, predict, prepare  # noqa: E402
from echo.viz import ACCENT, ACCENT2, FG, apply_style  # noqa: E402

GREEN = "#8fd694"


@dataclass
class Results:
    presence: dict[str, Any] = field(default_factory=dict)
    distance: dict[str, Any] = field(default_factory=dict)
    room: dict[str, Any] = field(default_factory=dict)
    room_session: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    verdict: list[str] = field(default_factory=list)


def _session_vote(fs: FrameSet, pred_room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Room per session by majority vote over its frames."""
    t, p = [], []
    for s in np.unique(fs.session):
        m = fs.session == s
        if fs.room[m][0] < 0:
            continue
        t.append(fs.room[m][0])
        p.append(np.bincount(pred_room[m], minlength=len(fs.rooms)).argmax())
    return np.asarray(t), np.asarray(p)


def evaluate_run(run_dir: Path, data_dir: Path | None = None, make_plots: bool = True) -> Results:
    split_json = json.loads((run_dir / "split.json").read_text())
    split = Split.from_json(split_json)
    data = data_dir or Path(split_json.get("data_dir", REPO_ROOT / "data/sessions"))
    prep = prepare(load_data(data), split)
    model, norm, rooms, ck = load_model(run_dir / "best.pt")
    torch.set_num_threads(4)
    test = prep.test

    cnn = predict(model, norm(test.x))
    energy = EnergyDetector(prep.bin_m).fit(prep.train)
    linear = LinearBaseline().fit(prep.train)
    lin = linear.predict(test)

    r = Results()
    r.info = {"run": run_dir.name, "data": str(data), "test_sessions": list(split.test),
              "train_sessions": len(split.train), "val_sessions": len(split.val),
              "group_by": split.group_by, "test_frames": len(test), "best_epoch": ck["epoch"] + 1,
              "rooms": rooms, "parameters": sum(p.numel() for p in model.parameters())}

    if np.isfinite(test.presence).any():
        r.presence["energy"] = binary_metrics(test.presence, energy.predict_proba(test)).as_dict()
        if np.isfinite(lin["presence"]).all():
            r.presence["logreg"] = binary_metrics(test.presence, lin["presence"]).as_dict()
        r.presence["cnn"] = binary_metrics(test.presence, cnn["presence"]).as_dict()

    present = (test.presence == 1) & np.isfinite(test.distance)
    if present.any():
        if np.isfinite(lin["distance"]).all():
            r.distance["ridge"] = regression_metrics(test.distance[present], lin["distance"][present]).as_dict()
        r.distance["cnn"] = regression_metrics(test.distance[present], cnn["distance"][present]).as_dict()

    if "room" in cnn and len(rooms) >= 2 and (test.room >= 0).any():
        r.room["logreg"] = class_metrics(test.room, lin["room"], rooms).as_dict()
        r.room["cnn"] = class_metrics(test.room, cnn["room"], rooms).as_dict()
        for name, pr in (("logreg", lin["room"]), ("cnn", cnn["room"])):
            t, p = _session_vote(test, pr)
            r.room_session[name] = class_metrics(t, p, rooms).as_dict()

    r.verdict = verdict(r)
    if make_plots:
        _plots(run_dir, prep.bin_m, test, cnn, energy.predict_proba(test), lin, r)
    (run_dir / "metrics.json").write_text(json.dumps(r.__dict__, indent=2, default=float))
    (run_dir / "results.md").write_text(results_markdown(r), encoding="utf-8")
    return r


def verdict(r: Results) -> list[str]:
    """Does the CNN beat the baselines? One line per task, with the reason."""
    out = []
    if r.presence:
        base = max((v["balanced_accuracy"], k) for k, v in r.presence.items() if k != "cnn")
        c = r.presence["cnn"]["balanced_accuracy"]
        out.append(f"presence: CNN balanced accuracy {c:.3f} vs best baseline ({base[1]}) {base[0]:.3f} -> "
                   + ("CNN better" if c > base[0] else "CNN NOT better"))
    if r.distance:
        c = r.distance["cnn"]["mae"]
        b = r.distance.get("ridge", {}).get("mae", float("nan"))
        out.append(f"distance: CNN MAE {c:.3f} m vs ridge {b:.3f} m -> "
                   + ("CNN better" if not c >= b else "CNN NOT better"))
    if r.room:
        c, b = r.room["cnn"]["accuracy"], r.room["logreg"]["accuracy"]
        out.append(f"room: CNN frame accuracy {c:.3f} vs logreg {b:.3f} -> "
                   + ("CNN better" if c > b else ("tie" if c == b else "CNN NOT better")))
    return out


def _fmt(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def results_markdown(r: Results) -> str:
    lines = [f"Held-out test: {len(r.info['test_sessions'])} sessions, {r.info['test_frames']} frames "
             f"(split by {r.info['group_by']}). CNN: {r.info['parameters']:,} parameters.", ""]
    if r.presence:
        lines += ["| Presence | accuracy | balanced acc. | F1 | AUROC |", "|---|---|---|---|---|"]
        for k, v in r.presence.items():
            lines.append(f"| {k} | {_fmt(v['accuracy'], True)} | {_fmt(v['balanced_accuracy'], True)} "
                         f"| {_fmt(v['f1'])} | {_fmt(v['auroc'])} |")
        lines.append("")
    if r.distance:
        lines += ["| Distance (person present) | MAE (m) | RMSE (m) | median (m) | within 25 cm |",
                  "|---|---|---|---|---|"]
        for k, v in r.distance.items():
            lines.append(f"| {k} | {_fmt(v['mae'])} | {_fmt(v['rmse'])} | {_fmt(v['median_ae'])} "
                         f"| {_fmt(v['within_25cm'], True)} |")
        lines.append("")
    if r.room:
        lines += ["| Room ID | frame accuracy | session accuracy (vote) |", "|---|---|---|"]
        for k, v in r.room.items():
            lines.append(f"| {k} | {_fmt(v['accuracy'], True)} "
                         f"| {_fmt(r.room_session.get(k, {}).get('accuracy'), True)} |")
        lines.append("")
    lines += [f"- {v}" for v in r.verdict]
    return "\n".join(lines) + "\n"


def _confusion(ax: Any, cm: list[list[int]], labels: list[str], title: str) -> None:
    m = np.asarray(cm, dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(m[i, j])}\n{100 * norm[i, j]:.0f}%", ha="center", va="center",
                    color="black" if norm[i, j] > 0.5 else FG, fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")


def _plots(run_dir: Path, bin_m: float, test: FrameSet, cnn: dict[str, np.ndarray],
           energy_p: np.ndarray, lin: dict[str, np.ndarray], r: Results) -> None:
    apply_style()
    panels = [p for p in (("presence", r.presence), ("room", r.room)) if p[1]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), constrained_layout=True,
                                 squeeze=False)
        for ax, (name, res) in zip(axes[0], panels):
            labels = ["absent", "present"] if name == "presence" else res["cnn"]["labels"]
            _confusion(ax, res["cnn"]["confusion"], labels, f"CNN {name} (test)")
        fig.savefig(run_dir / "confusion.png", dpi=130)
        plt.close(fig)

    present = (test.presence == 1) & np.isfinite(test.distance)
    if present.any():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        t = test.distance[present]
        lo, hi = float(t.min()) - 0.1, float(t.max()) + 0.1
        if np.isfinite(lin["distance"]).all():
            a1.scatter(t, lin["distance"][present], s=3, alpha=0.25, color=ACCENT2, label="ridge")
        a1.scatter(t, cnn["distance"][present], s=3, alpha=0.35, color=ACCENT, label="CNN")
        a1.plot([lo, hi], [lo, hi], color=FG, lw=0.8, ls="--")
        a1.set_xlabel("true distance (m)")
        a1.set_ylabel("predicted (m)")
        a1.set_title("prediction vs label", loc="left")
        a1.legend(frameon=False, markerscale=4)
        edges = np.arange(np.floor(lo * 4) / 4, hi + 0.25, 0.25)
        centres = (edges[:-1] + edges[1:]) / 2
        for name, pred, col in (("ridge", lin["distance"], ACCENT2), ("CNN", cnn["distance"], ACCENT)):
            if not np.isfinite(pred).all():
                continue
            err = np.abs(pred[present] - t)
            mae = [err[(t >= a) & (t < b)].mean() if ((t >= a) & (t < b)).any() else np.nan
                   for a, b in zip(edges[:-1], edges[1:])]
            a2.plot(centres, mae, marker="o", color=col, label=name)
        a2.set_xlabel("true distance (m)")
        a2.set_ylabel("mean abs. error (m)")
        a2.set_title("error vs distance", loc="left")
        a2.grid(True, alpha=0.4)
        a2.legend(frameon=False)
        fig.savefig(run_dir / "distance_error.png", dpi=130)
        plt.close(fig)

    # Timeline of the most "mixed" test session.
    best_s, best_score = None, -1.0
    for s in np.unique(test.session):
        m = test.session == s
        p = test.presence[m]
        p = p[np.isfinite(p)]
        score = min(p.mean(), 1 - p.mean()) if len(p) else -1.0
        score += 0.01 * np.isfinite(test.distance[m]).mean()
        if score > best_score:
            best_s, best_score = s, score
    if best_s is not None:
        m = test.session == best_s
        tt = np.arange(m.sum()) * 0.05
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, constrained_layout=True)
        a1.fill_between(tt, 0, np.nan_to_num(test.presence[m]), step="post", color=GREEN, alpha=0.25,
                        label="label")
        a1.plot(tt, energy_p[m], color=ACCENT2, lw=0.9, label="energy detector")
        a1.plot(tt, cnn["presence"][m], color=ACCENT, lw=1.1, label="CNN")
        a1.set_ylabel("P(person)")
        a1.set_ylim(-0.05, 1.05)
        a1.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
        a1.set_title(f"test session: {test.session_ids[best_s]}", loc="left")
        a2.plot(tt, test.distance[m], color=GREEN, lw=1.5, label="label")
        shown = np.where(cnn["presence"][m] >= 0.5, cnn["distance"][m], np.nan)
        a2.plot(tt, shown, color=ACCENT, lw=1.0, label="CNN (when present)")
        a2.set_ylabel("distance (m)")
        a2.set_xlabel("time (s)")
        a2.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
        a2.grid(True, alpha=0.4)
        fig.savefig(run_dir / "timeline.png", dpi=130)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="run")
    ap.add_argument("--data", type=Path, default=None, help="override the session directory")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = REPO_ROOT / cfg.train.out_dir / args.name
    r = evaluate_run(run_dir, args.data)
    print(results_markdown(r))
    print(f"Plots and metrics in {run_dir}")


if __name__ == "__main__":
    main()
