"""Small, dependency-free metrics (no scikit-learn)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    confusion: list[list[int]]  # [[tn, fp], [fn, tp]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    median_ae: float
    within_25cm: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClassMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    confusion: list[list[int]]
    labels: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties averaged)."""
    y = np.asarray(y).astype(bool)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    ok = np.isfinite(y_true)
    y = y_true[ok].astype(bool)
    s = y_score[ok]
    p = s >= threshold
    tp, tn = int(np.sum(p & y)), int(np.sum(~p & ~y))
    fp, fn = int(np.sum(p & ~y)), int(np.sum(~p & y))
    prec, rec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    bal = np.nanmean([rec, tnr]) if not (np.isnan(rec) and np.isnan(tnr)) else float("nan")
    f1 = _safe_div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else float("nan")
    return BinaryMetrics(int(ok.sum()), _safe_div(tp + tn, len(y)), float(bal), prec, rec, f1,
                         auroc(y, s), [[tn, fp], [fn, tp]])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return RegressionMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    e = np.abs(y_pred[ok] - y_true[ok])
    return RegressionMetrics(int(ok.sum()), float(e.mean()), float(np.sqrt(np.mean(e**2))),
                             float(np.median(e)), float(np.mean(e <= 0.25)))


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> ClassMetrics:
    ok = y_true >= 0
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true[ok], y_pred[ok]):
        cm[int(t), int(p)] += 1
    recalls = [cm[i, i] / cm[i].sum() for i in range(k) if cm[i].sum() > 0]
    return ClassMetrics(int(ok.sum()), _safe_div(np.trace(cm), cm.sum()),
                        float(np.mean(recalls)) if recalls else float("nan"), cm.tolist(), labels)


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (fit on training data only)."""
    ok = np.isfinite(y_true)
    y, s = y_true[ok].astype(bool), score[ok]
    if y.all() or (~y).all():
        return float(np.median(s))
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    best, best_t = -1.0, float(cands[0])
    for t in cands:
        p = s >= t
        bal = 0.5 * (np.mean(p[y]) + np.mean(~p[~y]))
        if bal > best:
            best, best_t = bal, float(t)
    return best_t
