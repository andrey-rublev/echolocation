"""Stage 0 acceptance: synthetic echo delays are recovered to within 1 sample."""

from __future__ import annotations

import numpy as np
import pytest

from echo.chirp import make_chirp
from echo.config import load_config
from echo.profile import envelope, find_echo_delays, matched_filter
from echo.simulate import Echo, add_impulse, simulate_recording


@pytest.fixture(scope="module")
def template() -> np.ndarray:
    cfg = load_config()
    return make_chirp(cfg.chirp, cfg.audio.sample_rate)


@pytest.mark.parametrize("seed", range(5))
def test_recovers_echo_delays_within_one_sample(template: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 4000
    direct = Echo(40.0, 1.0)
    # Well-separated echoes, integer and fractional delays, decreasing amplitude.
    delays = np.sort(rng.uniform(150, 1100, size=4))
    while np.min(np.diff(delays)) < 60:
        delays = np.sort(rng.uniform(150, 1100, size=4))
    echoes = [Echo(float(d), float(a)) for d, a in zip(delays, [0.3, 0.15, 0.08, 0.05])]
    rec = simulate_recording(template, echoes, n, direct=direct, noise_std=0.01, rng=rng)

    env = envelope(matched_filter(rec, template))
    found = find_echo_delays(env, n=5, min_separation=40)
    assert len(found) == 5
    expected = sorted([direct.delay_samples, *delays])
    np.testing.assert_allclose(found, expected, atol=1.0)


def test_direct_path_is_strongest(template: np.ndarray) -> None:
    rec = simulate_recording(template, [Echo(500, 0.2)], 3000, direct=Echo(25, 1.0), noise_std=0.005)
    env = envelope(matched_filter(rec, template))
    assert abs(int(np.argmax(env)) - 25) <= 1


def test_integer_impulse_is_exact() -> None:
    h = np.zeros(50)
    add_impulse(h, 17.0, 0.5)
    assert h[17] == 0.5 and np.count_nonzero(h) == 1


def test_fractional_impulse_is_centred() -> None:
    h = np.zeros(80)
    add_impulse(h, 30.25, 1.0)
    centroid = np.sum(np.arange(80) * h) / np.sum(h)
    assert abs(centroid - 30.25) < 0.05


def test_noise_only_has_no_strong_peak(template: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    rec = simulate_recording(template, [], 3000, direct=None, noise_std=0.01, rng=rng)
    env = envelope(matched_filter(rec, template))
    # Matched-filter noise floor: std * sqrt(template energy); stays well below a real echo.
    floor = 0.01 * np.sqrt(np.sum(template**2))
    assert env.max() < 6 * floor
