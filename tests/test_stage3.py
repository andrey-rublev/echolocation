"""Stage 3 (simulator): labels, session I/O, collection, camera maths, splits."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from echo.camera import PoseSample, align_labels, body_size_px, distance_from_size, BodySize
from echo.collect import collect_session
from echo.config import load_config
from echo.dataset import (
    Session,
    StreamingFeatures,
    load_session,
    parse_label,
    session_features,
    split_sessions,
)
from echo.sonar import SimScene, open_source
from echo.viz import sessions_figure


@pytest.mark.parametrize("label,expected", [
    ("empty", (0.0, math.nan)),
    ("person", (1.0, math.nan)),
    ("walking", (1.0, math.nan)),
    ("person_1m", (1.0, 1.0)),
    ("person_1.5m", (1.0, 1.5)),
    ("person_150cm", (1.0, 1.5)),
    ("room=dorm_desk", (math.nan, math.nan)),
])
def test_parse_label(label: str, expected: tuple[float, float]) -> None:
    got = parse_label(label)
    np.testing.assert_equal(got, expected)


def _collect(tmp: Path, person: str, label: str, camera: bool, seconds: float = 6.0,
             room: str = "sim_room", minute: int = 0) -> Path:
    cfg = load_config()
    src = open_source(cfg, sim=True, scene=SimScene(room=room, person=person, realtime=False))
    return collect_session(cfg, src, room, label, seconds, tmp, camera=camera, progress=False,
                           when=datetime(2026, 3, 1, 12, minute))


def test_collect_manual_session_roundtrip(tmp_path: Path) -> None:
    p = _collect(tmp_path, "static:1.2", "person_1.2m", camera=False)
    s = load_session(p)
    assert p.with_suffix(".json").exists()
    assert len(s) == 120 and s.profiles.shape == (120, s.grid.n_bins)
    assert np.all(s.presence == 1) and np.allclose(s.distance, 1.2)
    for key in ("room", "device", "gain", "date", "notes", "label_source", "config"):
        assert key in s.meta
    assert s.meta["label_source"] == "manual" and s.meta["simulated"] is True
    assert np.all(np.diff(s.times) > 0)


def test_collect_camera_session_labels_follow_truth(tmp_path: Path) -> None:
    p = _collect(tmp_path, "linear:0.8:2.8:6", "auto", camera=True)
    s = load_session(p)
    assert s.meta["label_source"] == "camera"
    assert np.all(s.presence == 1)
    d = s.distance
    assert d[-1] > d[0] + 1.0  # moving away
    assert np.all((d > 0.6) & (d < 3.2))


def test_align_labels_nearest_and_stale() -> None:
    samples = [PoseSample(10.0, True, 1.0), PoseSample(10.1, False, math.nan),
               PoseSample(10.2, True, 2.0)]
    times = np.array([9.0, 10.02, 10.09, 10.24, 11.0])
    pres, dist = align_labels(times, samples, max_age_s=0.1)
    np.testing.assert_equal(pres, [np.nan, 1, 0, 1, np.nan])
    np.testing.assert_equal(dist, [np.nan, 1.0, np.nan, 2.0, np.nan])


def test_body_size_and_distance() -> None:
    def lm(x: float, y: float, v: float = 0.9) -> SimpleNamespace:
        return SimpleNamespace(x=x, y=y, visibility=v)

    lms = [lm(0, 0)] * 33
    lms = list(lms)
    lms[11], lms[12] = lm(0.40, 0.30), lm(0.60, 0.30)  # shoulders 0.2 * 640 = 128 px
    lms[23], lms[24] = lm(0.42, 0.65), lm(0.58, 0.65)  # torso 0.35 * 480 = 168 px
    size = body_size_px(lms, 640, 480, 0.5)
    assert size.shoulder_px == pytest.approx(128.0)
    assert size.torso_px == pytest.approx(168.0)
    # K = 230 px*m: shoulder estimate 1.797 m; torso estimate 230*1.3/168 = 1.780 m -> min.
    assert distance_from_size(size, 230.0, 299.0) == pytest.approx(299.0 / 168.0)
    lms[23] = lm(0.42, 0.65, v=0.1)  # hip hidden -> shoulder only
    size2 = body_size_px(lms, 640, 480, 0.5)
    assert math.isnan(size2.torso_px)
    assert distance_from_size(size2, 230.0, 299.0) == pytest.approx(230.0 / 128.0)
    assert math.isnan(distance_from_size(BodySize(math.nan, math.nan), 230.0, 299.0))


def test_streaming_features_match_batch(tmp_path: Path) -> None:
    s = load_session(_collect(tmp_path, "wave", "person", camera=False, seconds=3))
    batch = session_features(s)
    sf = StreamingFeatures()
    live = np.stack([sf.update(p, a) for p, a in zip(s.profiles, s.direct_amp)])
    np.testing.assert_allclose(live, batch, rtol=1e-4, atol=1e-4)


def _fake_session(sid: str, room: str, day: str, presence: float) -> Session:
    n = 10
    return Session(Path(sid), np.zeros((n, 4), np.float32), np.arange(n, dtype=float),
                   np.ones(n), np.zeros(n), np.full(n, presence), np.full(n, np.nan),
                   {"session_id": sid, "room": room, "day": day})


def test_split_never_shares_sessions_or_days() -> None:
    sessions = [_fake_session(f"s{i}", f"r{i % 2}", f"2026-01-0{1 + i // 4}", float(i % 2 == 0))
                for i in range(12)]
    sp = split_sessions(sessions, test_fraction=0.3, val_fraction=0.2, seed=1)
    assert sp.group_by == "day"
    all_ids = sp.train + sp.val + sp.test
    assert len(all_ids) == len(set(all_ids)) == 12
    day_of = {s.id: s.day for s in sessions}
    assert not ({day_of[i] for i in sp.test} & {day_of[i] for i in sp.train})
    # Single day -> grouped by session, stratified so test has both kinds.
    one_day = [_fake_session(f"t{i}", "r", "2026-01-01", float(i % 2)) for i in range(10)]
    sp2 = split_sessions(one_day, seed=0)
    assert sp2.group_by == "session"
    kinds = {one_day[int(i[1:])].presence[0] for i in sp2.test}
    assert kinds == {0.0, 1.0}
    assert not set(sp2.test) & set(sp2.train)


def test_sessions_figure(tmp_path: Path) -> None:
    a = _collect(tmp_path, "none", "empty", camera=False, seconds=4, minute=1)
    b = _collect(tmp_path, "walk", "auto", camera=True, seconds=4, minute=2)
    fig = sessions_figure([a, b], alpha=0.05)
    out = tmp_path / "s.png"
    fig.savefig(out)
    assert out.stat().st_size > 10_000
