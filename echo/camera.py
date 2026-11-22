"""Webcam auto-labels: is a person there, and roughly how far away?

MediaPipe Pose Landmarker finds the body; distance comes from a pinhole model.
An object of real size W at distance d appears ``w = f * W / d`` pixels wide,
so ``d = K / w`` with ``K = f * W``. ``K`` is measured once per person and
camera by standing at a known distance (``calibrate``).

Two body sizes are used and the *smaller* distance estimate wins:

* shoulder width (both shoulders visible) - shrinks when you turn sideways,
* torso height (shoulder midpoint to hip midpoint) - robust to turning but
  needs the hips in frame (typically beyond ~1.2 m).

Turning or partial occlusion only ever makes a body part look *smaller*
(farther), so the minimum is the safer estimate.

CLI::

    python -m echo.camera download-model          # fetch the MediaPipe model once
    python -m echo.camera preview                 # live view with distance overlay
    python -m echo.camera calibrate --distance 1.5
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from echo.config import DEFAULT_CONFIG_PATH, REPO_ROOT, CameraConfig, load_config, update_yaml_value

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# MediaPipe pose landmark indices.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


@dataclass(frozen=True)
class PoseSample:
    t: float  # time.monotonic() when the camera frame was read
    present: bool
    distance_m: float  # NaN when absent or not measurable
    shoulder_px: float = math.nan
    torso_px: float = math.nan


@dataclass(frozen=True)
class BodySize:
    shoulder_px: float  # NaN if not visible
    torso_px: float


def body_size_px(landmarks: Any, width: int, height: int, min_visibility: float) -> BodySize:
    """Shoulder width and torso height in pixels from normalised landmarks."""

    def pt(i: int) -> tuple[float, float] | None:
        lm = landmarks[i]
        vis = getattr(lm, "visibility", 1.0)
        if vis is not None and vis < min_visibility:
            return None
        return lm.x * width, lm.y * height

    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)
    shoulder = math.dist(ls, rs) if ls and rs else math.nan
    torso = math.nan
    if ls and rs and lh and rh:
        sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        torso = math.dist(sm, hm)
    return BodySize(shoulder, torso)


def distance_from_size(size: BodySize, k_shoulder: float, k_torso: float | None) -> float:
    """Pinhole distance; the smaller of the available estimates (see module doc)."""
    cands = []
    if np.isfinite(size.shoulder_px) and size.shoulder_px > 1:
        cands.append(k_shoulder / size.shoulder_px)
    if k_torso is not None and np.isfinite(size.torso_px) and size.torso_px > 1:
        cands.append(k_torso / size.torso_px)
    return float(min(cands)) if cands else math.nan


def torso_k(cfg: CameraConfig) -> float:
    """Torso-height constant. An adult torso (shoulders to hips) is ~1.2x the
    shoulder width; ``calibrate`` stores the measured ratio in the config."""
    return cfg.shoulder_k_px_m * cfg.torso_to_shoulder


class PoseEstimator:
    """Thin wrapper around MediaPipe's Pose Landmarker in VIDEO mode."""

    def __init__(self, cfg: CameraConfig) -> None:
        import mediapipe as mp

        model = Path(cfg.model_path)
        model = model if model.is_absolute() else REPO_ROOT / model
        if not model.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model}. Run `python -m echo.camera download-model` "
                f"(downloads {MODEL_URL})."
            )
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1
        self.cfg = cfg

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, t: float) -> tuple[PoseSample, Any]:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        if not result.pose_landmarks:
            return PoseSample(t, False, math.nan), None
        lms = result.pose_landmarks[0]
        h, w = frame_bgr.shape[:2]
        size = body_size_px(lms, w, h, self.cfg.min_visibility)
        d = distance_from_size(size, self.cfg.shoulder_k_px_m, torso_k(self.cfg))
        return PoseSample(t, True, d, size.shoulder_px, size.torso_px), lms


@dataclass
class CameraLabeler:
    """Background thread: webcam frames -> timestamped :class:`PoseSample` list."""

    cfg: CameraConfig
    samples: list[PoseSample] = field(default_factory=list)
    error: BaseException | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _ready: threading.Event = field(default_factory=threading.Event)

    def _run(self) -> None:
        import cv2

        cap = None
        est = None
        try:
            est = PoseEstimator(self.cfg)
            cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam {self.cfg.device}.")
            self._ready.set()
            while not self._stop.is_set():
                ok, frame = cap.read()
                t = time.monotonic()
                if not ok:
                    continue
                sample, _ = est.detect(frame, t)
                self.samples.append(sample)
        except BaseException as exc:
            self.error = exc
            self._ready.set()
        finally:
            if cap is not None:
                cap.release()
            if est is not None:
                est.close()

    def start(self, timeout_s: float = 15.0) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout_s)
        if self.error is not None:
            raise RuntimeError(f"Camera failed to start: {self.error}") from self.error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def align_labels(frame_times: np.ndarray, samples: list[PoseSample],
                 max_age_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest camera sample for every audio frame (both on time.monotonic()).

    Frames with no camera sample within ``max_age_s`` get NaN labels.
    """
    n = len(frame_times)
    presence = np.full(n, np.nan, dtype=np.float32)
    distance = np.full(n, np.nan, dtype=np.float32)
    if not samples:
        return presence, distance
    st = np.array([s.t for s in samples])
    order = np.argsort(st)
    st = st[order]
    sp = np.array([s.present for s in samples], dtype=np.float32)[order]
    sd = np.array([s.distance_m for s in samples], dtype=np.float32)[order]
    j = np.clip(np.searchsorted(st, frame_times), 1, len(st) - 1) if len(st) > 1 else np.zeros(n, int)
    if len(st) > 1:
        left_closer = np.abs(frame_times - st[j - 1]) <= np.abs(frame_times - st[j])
        j = np.where(left_closer, j - 1, j)
    ok = np.abs(st[j] - frame_times) <= max_age_s
    presence[ok] = sp[j[ok]]
    distance[ok] = np.where(sp[j[ok]] > 0, sd[j[ok]], np.nan)
    return presence, distance


def sim_camera_labels(frame_times: np.ndarray, truth: Callable[[float], float | None],
                      rng: np.random.Generator, rel_noise: float = 0.05,
                      max_range_m: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    """What the camera labeller would produce in the simulator (noisy ground truth)."""
    pres = np.zeros(len(frame_times), np.float32)
    dist = np.full(len(frame_times), np.nan, np.float32)
    for i, t in enumerate(frame_times):
        r = truth(float(t))
        if r is not None and r < max_range_m:
            pres[i] = 1.0
            dist[i] = r * (1.0 + rng.normal(0.0, rel_noise))
    return pres, dist


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _download_model(cfg: CameraConfig) -> None:
    dest = Path(cfg.model_path)
    dest = dest if dest.is_absolute() else REPO_ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}\n -> {dest}")
    urllib.request.urlretrieve(MODEL_URL, dest)  # noqa: S310 (fixed official URL)
    print(f"Done ({dest.stat().st_size / 1e6:.1f} MB)")


def _preview(cfg: CameraConfig, seconds: float | None) -> list[PoseSample]:
    import cv2

    est = PoseEstimator(cfg)
    cap = cv2.VideoCapture(cfg.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {cfg.device}.")
    samples: list[PoseSample] = []
    start = time.monotonic()
    try:
        while seconds is None or time.monotonic() - start < seconds:
            ok, frame = cap.read()
            t = time.monotonic()
            if not ok:
                continue
            s, lms = est.detect(frame, t)
            samples.append(s)
            h, w = frame.shape[:2]
            if lms is not None:
                for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
                    cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 6, (94, 196, 255), -1)
            text = (f"person  d={s.distance_m:.2f} m  shoulders={s.shoulder_px:.0f}px  "
                    f"torso={s.torso_px:.0f}px" if s.present else "no person")
            cv2.rectangle(frame, (0, 0), (w, 34), (17, 14, 14), -1)
            cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (233, 222, 216), 2)
            cv2.imshow("echolocation camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        est.close()
        cv2.destroyAllWindows()
    return samples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download-model", help="download the MediaPipe pose model")
    pv = sub.add_parser("preview", help="live webcam view with distance estimate")
    pv.add_argument("--seconds", type=float, default=None)
    cal = sub.add_parser("calibrate", help="stand at a known distance to measure K")
    cal.add_argument("--distance", type=float, required=True, help="your distance from the camera (m)")
    cal.add_argument("--seconds", type=float, default=6.0)
    cal.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    path = args.config or DEFAULT_CONFIG_PATH
    cfg = load_config(path).camera
    if args.cmd == "download-model":
        _download_model(cfg)
        return 0
    if args.cmd == "preview":
        _preview(cfg, args.seconds)
        return 0
    print(f"Stand facing the camera, shoulders square, {args.distance:.2f} m away. "
          f"Measuring for {args.seconds:.0f} s (press q to stop early)...")
    samples = [s for s in _preview(cfg, args.seconds) if s.present]
    sh = np.array([s.shoulder_px for s in samples])
    to = np.array([s.torso_px for s in samples])
    sh, to = sh[np.isfinite(sh)], to[np.isfinite(to)]
    if len(sh) < 10:
        print("Not enough frames with both shoulders visible; check lighting and framing.")
        return 1
    k = float(np.median(sh)) * args.distance
    print(f"Median shoulder width {np.median(sh):.1f} px -> shoulder_k_px_m = {k:.1f}")
    ratio = None
    if len(to) >= 10:
        ratio = float(np.median(to) / np.median(sh))
        print(f"Median torso height {np.median(to):.1f} px -> torso_to_shoulder = {ratio:.3f}")
    else:
        print("Hips not visible: keeping the default torso_to_shoulder ratio "
              "(stand farther back to calibrate it).")
    if not args.dry_run:
        update_yaml_value(path, "camera", "shoulder_k_px_m", round(k, 2))
        if ratio is not None:
            update_yaml_value(path, "camera", "torso_to_shoulder", round(ratio, 4))
        print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
