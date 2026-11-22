"""Stage 1 checks: stream timing/jitter and amplitude linearity.

CLI::

    python -m echo.diagnostics timing --seconds 10 [--save-wav] [--sim]
    python -m echo.diagnostics linearity [--sim [--sim-agc]]

``timing`` confirms ~20 chirps/s are captured with a stable direct-path
arrival. ``linearity`` alternates two gains inside one stream and checks the
recorded direct-path amplitude scales with them and does not drift; if either
fails, Windows audio processing (AGC, noise suppression) is probably on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from echo.config import Config
from echo.profile import EchoFrame, EchoProcessor, range_grid
from echo.sonar import Source, add_source_args, source_from_args, stream_frames

WINDOWS_AUDIO_HELP = """\
Windows audio processing is probably altering the signal. Turn these off, then re-run:

  1. Settings > System > Sound > Input > (your microphone) > Properties:
       - "Audio enhancements" -> Off
       - Studio effects / "Voice focus" (Windows Studio Effects on Core Ultra NPUs) -> Off
  2. Settings > System > Sound > Output > (your speakers) > Properties:
       - "Audio enhancements" -> Off, spatial sound -> Off
  3. Classic control panel (Win+R, mmsys.cpl):
       - Recording > Microphone Array > Properties > Enhancements: tick "Disable all sound effects"
       - Recording > ... > Levels: fixed level (~70-80), Microphone Boost 0 dB
       - Playback > Speakers > Properties > Enhancements: "Disable all sound effects"
       - Communications tab: "Do nothing" (stops ducking)
  4. Acer software: AcerSense / Acer PurifiedVoice -> turn off AI noise reduction / PurifiedVoice;
     Acer TrueHarmony / DTS / Dolby audio -> Off.
  5. Realtek Audio Console or Intel Smart Sound (if present): noise suppression,
     beam forming, echo cancellation -> Off.
  6. Close Teams/Zoom/Discord (they enable echo cancellation on the shared device).
  7. Still failing? Set audio.wasapi_exclusive: true in echo/config.yaml (bypasses the shared mixer).
"""


def _banner(msg: str) -> str:
    bar = "!" * 78
    return f"\n{bar}\n!! {msg}\n{bar}\n"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@dataclass
class TimingResult:
    seconds: float
    periods_expected: int
    chirps_detected: int
    profiles: int
    lock_phase: int | None
    lock_spread: float
    jitter_std: float
    jitter_p2p: float
    direct_amp_median: float
    direct_to_floor_db: float
    xruns: int
    peak_abs: float
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    def report(self) -> str:
        lines = [
            f"Run length        : {self.seconds:.1f} s ({self.periods_expected} chirp periods)",
            f"Chirps detected   : {self.chirps_detected}  (profiles after lock: {self.profiles})",
            f"Direct-path phase : {self.lock_phase} samples (warm-up spread {self.lock_spread:.0f})",
            f"Direct-path jitter: std {self.jitter_std:.2f}, peak-to-peak {self.jitter_p2p:.2f} samples",
            f"Direct amplitude  : {self.direct_amp_median:.3f} (direct/floor {self.direct_to_floor_db:.1f} dB)",
            f"Input peak        : {self.peak_abs:.3f} of full scale",
            f"Xruns             : {self.xruns}",
            "RESULT            : " + ("PASS" if self.passed else "FAIL: " + "; ".join(self.problems)),
        ]
        return "\n".join(lines)


def _floor(frames: list[EchoFrame], cfg: Config) -> float:
    grid = range_grid(cfg)
    far = grid.axis_m() > 0.75 * cfg.profile.max_range_m
    return float(np.median(np.stack([f.profile[far] for f in frames]))) + 1e-12


def run_timing(cfg: Config, source: Source, seconds: float = 10.0) -> TimingResult:
    P = cfg.period_samples
    periods = int(round(seconds * cfg.audio.sample_rate / P))
    peak = [0.0]

    def track(b: np.ndarray) -> None:
        peak[0] = max(peak[0], float(np.max(np.abs(b))) if len(b) else 0.0)

    frames: list[EchoFrame] = []
    # A dedicated processor (no automatic re-lock) so xruns show up as jitter.
    proc: EchoProcessor | None = None
    consumed = 0
    with source:
        for block in source.blocks():
            track(block)
            if proc is None:
                proc = EchoProcessor(cfg, t0=source.t0)
            frames.extend(proc.push(block))
            consumed += len(block)
            if consumed >= periods * P + 2 * P:
                break
    frames = [f for f in frames if f.index < periods]
    problems: list[str] = []
    lock = proc.lock if proc is not None else None
    if not frames or lock is None:
        return TimingResult(seconds, periods, 0, 0, None, float("nan"), float("nan"), float("nan"),
                            0.0, float("-inf"), source.xruns, peak[0],
                            ["no chirps detected: check volume, mute switches and device ids"])
    lags = np.array([f.direct_lag for f in frames])
    amps = np.array([f.direct_amp for f in frames])
    floor = _floor(frames, cfg)
    med = float(np.median(amps))
    # Warm-up periods are chirps too; they were used for locking, not profiled.
    detected = int(np.sum(amps > 0.25 * med)) + lock.n_obs
    p2p = float(np.ptp(lags))
    if detected < 0.9 * periods:
        problems.append(f"only {detected}/{periods} chirps detected")
    if p2p > 2.0:
        problems.append(f"direct-path jitter {p2p:.1f} samples > 2")
    if source.xruns:
        problems.append(f"{source.xruns} xruns (buffer over/underflow); try audio.latency: high")
    if 20 * np.log10(med / floor) < 20:
        problems.append("direct path barely above the noise floor: raise volume or check mic")
    if peak[0] > 0.98:
        problems.append("input is clipping: lower --gain or the mic level")
    return TimingResult(
        seconds=seconds,
        periods_expected=periods,
        chirps_detected=min(detected, periods),
        profiles=len(frames),
        lock_phase=lock.phase,
        lock_spread=lock.spread,
        jitter_std=float(np.std(lags)),
        jitter_p2p=p2p,
        direct_amp_median=med,
        direct_to_floor_db=float(20 * np.log10(med / floor)),
        xruns=source.xruns,
        peak_abs=peak[0],
        problems=problems,
    )


# --------------------------------------------------------------------------- #
# Linearity
# --------------------------------------------------------------------------- #
@dataclass
class LinearityResult:
    gains: list[float]
    amplitudes: list[float]  # median direct-path amplitude per step
    ratios: list[float]  # measured amplitude ratio between consecutive hi/lo steps
    expected_ratio: float
    drift: float  # relative spread of same-gain steps
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    def report(self) -> str:
        lines = ["step  gain    direct amp"]
        for i, (g, a) in enumerate(zip(self.gains, self.amplitudes)):
            lines.append(f"{i:>4}  {g:.3f}   {a:.4f}")
        lines.append(f"hi/lo ratios: {', '.join(f'{r:.2f}' for r in self.ratios)} "
                     f"(expected {self.expected_ratio:.2f})")
        lines.append(f"same-gain drift: {100 * self.drift:.1f}%")
        lines.append("RESULT: " + ("PASS (recording scales linearly)" if self.passed
                                   else "FAIL: " + "; ".join(self.problems)))
        return "\n".join(lines)


def run_linearity(
    cfg: Config,
    source: Source,
    high_gain: float | None = None,
    step_seconds: float = 2.0,
    steps: int = 6,
    settle_seconds: float = 0.6,
    tolerance: float = 0.15,
) -> LinearityResult:
    """Alternate between ``high_gain`` and half of it within one stream."""
    hi = source.gain if high_gain is None else high_gain
    lo = hi / 2
    gains = [hi if i % 2 == 0 else lo for i in range(steps)]
    per_step = int(round(step_seconds / cfg.chirp.period_s))
    settle = int(round(settle_seconds / cfg.chirp.period_s))
    amps: list[list[float]] = [[] for _ in gains]
    with source:
        source.set_gain(gains[0])
        count = 0
        for f in stream_frames(source, cfg):
            step = count // per_step
            if step >= steps:
                break
            if count % per_step == 0:
                source.set_gain(gains[step])
            if count % per_step >= settle:
                amps[step].append(f.direct_amp)
            count += 1
        source.set_gain(hi)
    med = [float(np.median(a)) if a else 0.0 for a in amps]
    ratios = [med[i] / med[i + 1] for i in range(0, steps - 1, 2) if med[i + 1] > 0]
    expected = hi / lo
    his, los = np.array(med[0::2]), np.array(med[1::2])
    drift = float(max(np.ptp(his) / np.mean(his), np.ptp(los) / np.mean(los))) if med[1] > 0 else 1.0
    problems = []
    bad = [r for r in ratios if abs(r / expected - 1) > tolerance]
    if bad or not ratios:
        problems.append(f"amplitude ratio {', '.join(f'{r:.2f}' for r in ratios)} != {expected:.2f}")
    if drift > 0.10:
        problems.append(f"same-gain amplitude drifts by {100 * drift:.0f}% (adaptive processing?)")
    return LinearityResult(gains, med, ratios, expected, drift, problems)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timing", help="10 s run: chirp count and direct-path jitter")
    add_source_args(t)
    t.add_argument("--seconds", type=float, default=10.0)
    t.add_argument("--save-wav", action="store_true", help="save capture/playback WAVs to debug/")
    lin = sub.add_parser("linearity", help="check recorded amplitude scales with gain")
    add_source_args(lin)
    lin.add_argument("--step-seconds", type=float, default=2.0)
    args = ap.parse_args(argv)

    cfg, source = source_from_args(args, realtime=False, keep_raw=getattr(args, "save_wav", False))
    print(f"Source: {source.device_name}, gain {source.gain:.3f}")
    if args.cmd == "timing":
        res = run_timing(cfg, source, args.seconds)
        print(res.report())
        if args.save_wav and not args.sim:
            from echo.audio_io import debug_dir, save_wavs

            out = save_wavs(debug_dir(), cfg.audio.sample_rate, source.raw_capture,  # type: ignore[attr-defined]
                            source.raw_playback)  # type: ignore[attr-defined]
            print(f"WAVs saved to {out}")
        if not res.passed:
            print(_banner("TIMING CHECK FAILED"))
            if any("jitter" in p or "detected" in p for p in res.problems):
                print(WINDOWS_AUDIO_HELP)
        return 0 if res.passed else 1
    res_l = run_linearity(cfg, source, step_seconds=args.step_seconds)
    print(res_l.report())
    if not res_l.passed:
        print(_banner("LINEARITY CHECK FAILED: Windows audio enhancements / AGC / noise "
                      "suppression are probably still ON"))
        print(WINDOWS_AUDIO_HELP)
    return 0 if res_l.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
