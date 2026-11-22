from __future__ import annotations

import numpy as np

from echo.chirp import PeriodicPlayer, chirp_train, make_chirp, make_period
from echo.config import load_config


def test_chirp_shape_and_taper() -> None:
    cfg = load_config()
    ch = make_chirp(cfg.chirp, cfg.audio.sample_rate)
    assert len(ch) == cfg.chirp_samples == 960
    assert np.isclose(np.max(np.abs(ch)), 1.0)
    # Tapered ends: no click at start or end.
    assert abs(ch[0]) < 1e-6 and abs(ch[-1]) < 1e-3
    assert np.max(np.abs(np.diff(ch[:20]))) < 0.05


def test_chirp_spectrum_in_band() -> None:
    cfg = load_config()
    fs = cfg.audio.sample_rate
    ch = make_chirp(cfg.chirp, fs)
    spec = np.abs(np.fft.rfft(ch, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, 1 / fs)
    band = (freqs >= cfg.chirp.f0_hz) & (freqs <= cfg.chirp.f1_hz)
    assert spec[band].sum() / spec.sum() > 0.95


def test_period_and_train() -> None:
    cfg = load_config()
    fs = cfg.audio.sample_rate
    per = make_period(cfg.chirp, fs)
    assert len(per) == cfg.period_samples
    assert np.all(per[cfg.chirp_samples :] == 0)
    tr = chirp_train(cfg.chirp, fs, 3, gain=0.1)
    assert tr.dtype == np.float32 and len(tr) == 3 * len(per)
    assert np.isclose(np.max(np.abs(tr)), 0.1, atol=1e-6)


def test_periodic_player_matches_train_for_any_block_size() -> None:
    cfg = load_config()
    fs = cfg.audio.sample_rate
    per = make_period(cfg.chirp, fs)
    player = PeriodicPlayer(per, gain=0.1)
    blocks = [player.fill(n) for n in (100, 2400, 7, 4893, 1000)]
    got = np.concatenate(blocks)
    expected = chirp_train(cfg.chirp, fs, 6, gain=0.1)[: len(got)]
    np.testing.assert_allclose(got, expected)
