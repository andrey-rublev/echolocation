"""Range calibration with a hard flat object at a known distance.

    python -m echo.calibrate --distance 0.50

1. With the area in front of the laptop clear, press Enter (records a reference).
2. Place a hard flat object (a book, standing upright, facing the laptop) at the
   given distance from the speaker/mic edge, press Enter.
3. The object's echo is found in the difference of the two mean profiles, and
   ``profile.calib_offset_samples`` is updated in config.yaml so the measured
   distance matches the true one.

``--sim`` runs the same procedure on the simulator (object = static target).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

from echo.config import DEFAULT_CONFIG_PATH, Config, load_config, update_yaml_value
from echo.profile import meters_to_delay, parabolic_peak, range_grid, stack_profiles
from echo.sonar import SimScene, Source, add_source_args, open_source, stream_frames


@dataclass(frozen=True)
class CalibrationResult:
    true_m: float
    measured_m: float
    old_offset: float
    new_offset: float
    peak_db: float  # object echo above the residual floor

    @property
    def error_m(self) -> float:
        return self.measured_m - self.true_m

    def report(self) -> str:
        return (
            f"True distance     : {self.true_m:.3f} m\n"
            f"Measured distance : {self.measured_m:.3f} m (error {100 * self.error_m:+.1f} cm)\n"
            f"Echo strength     : {self.peak_db:.1f} dB above residual floor\n"
            f"Offset            : {self.old_offset:.2f} -> {self.new_offset:.2f} samples"
        )


def mean_profile(cfg: Config, source: Source, seconds: float) -> np.ndarray:
    n = int(round(seconds / cfg.chirp.period_s))
    with source:
        frames = list(stream_frames(source, cfg, max_frames=n))
    if not frames:
        raise RuntimeError("No echo frames captured.")
    P = stack_profiles(frames)
    amps = np.array([f.direct_amp for f in frames])
    return (P / amps[:, None]).mean(axis=0)  # normalise out any small gain drift


def estimate(cfg: Config, reference: np.ndarray | None, with_object: np.ndarray, true_m: float,
             search_m: float = 0.3) -> CalibrationResult:
    grid = range_grid(cfg)
    axis = grid.axis_m()
    diff = with_object - reference if reference is not None else with_object
    sig = np.abs(diff)
    window = (axis > true_m - search_m) & (axis < true_m + search_m)
    if not window.any():
        raise ValueError("Search window is outside the configured range.")
    idx = np.flatnonzero(window)
    i = int(idx[np.argmax(sig[idx])])
    # Sub-bin peak, converted back to metres on the bin-centre axis.
    pos = parabolic_peak(sig, i)
    measured = float(np.interp(pos, np.arange(len(axis)), axis))
    floor = float(np.median(sig[~window])) + 1e-12
    old = cfg.profile.calib_offset_samples
    new = old + float(meters_to_delay(measured - true_m, cfg.audio.sample_rate,
                                      cfg.profile.speed_of_sound_mps))
    return CalibrationResult(true_m, measured, old, new, float(20 * np.log10(sig[i] / floor)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--distance", type=float, required=True, help="true object distance (m)")
    ap.add_argument("--seconds", type=float, default=3.0, help="recording length per step")
    ap.add_argument("--search", type=float, default=0.3, help="search +/- this many metres")
    ap.add_argument("--no-reference", action="store_true", help="skip the empty-scene reference")
    ap.add_argument("--dry-run", action="store_true", help="do not write config.yaml")
    args = ap.parse_args(argv)
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = load_config(cfg_path)

    res = calibrate(cfg, args.distance, sim=args.sim, seconds=args.seconds, search_m=args.search,
                    use_reference=not args.no_reference, gain=args.gain,
                    allow_loud=args.allow_loud, sim_room=args.sim_room, interactive=not args.sim)
    print(res.report())
    if res.peak_db < 6:
        print("WARNING: the object's echo is weak; move it closer, use a larger flat object, "
              "or raise the gain a little.")
    if not args.dry_run:
        update_yaml_value(cfg_path, "profile", "calib_offset_samples", round(res.new_offset, 2))
        print(f"Saved calib_offset_samples to {cfg_path}")
    return 0


def calibrate(
    cfg: Config,
    distance_m: float,
    sim: bool,
    seconds: float = 3.0,
    search_m: float = 0.3,
    use_reference: bool = True,
    gain: float | None = None,
    allow_loud: bool = False,
    sim_room: str = "sim_room",
    interactive: bool = True,
) -> CalibrationResult:
    # Full-resolution profiles (no decimation) so one pass gives sub-sample accuracy.
    cfg_cal = replace(cfg, profile=replace(cfg.profile, decimate=1))

    def src(person: str) -> Source:
        scene = SimScene(room=sim_room, person=person, realtime=False)
        return open_source(cfg_cal, sim, gain=gain, allow_loud=allow_loud, scene=scene)

    reference = None
    if use_reference:
        if interactive:
            input("Clear the space in front of the laptop, then press Enter...")
        reference = mean_profile(cfg_cal, src("none"), seconds)
    if interactive:
        input(f"Place the flat object {distance_m:.2f} m from the laptop, then press Enter...")
    with_obj = mean_profile(cfg_cal, src(f"book:{distance_m}"), seconds)
    return estimate(cfg_cal, reference, with_obj, distance_m, search_m)


if __name__ == "__main__":
    raise SystemExit(main())
