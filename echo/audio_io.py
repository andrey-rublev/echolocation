"""Audio devices and the full-duplex sonar stream.

``sounddevice`` is imported lazily so the rest of the package (and the test
suite) works on machines without PortAudio or audio hardware.

CLI::

    python -m echo.audio_io list               # show devices, WASAPI first
    python -m echo.audio_io run --seconds 10   # duplex run, saves WAVs to debug/
"""

from __future__ import annotations

import argparse
import queue
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from scipy.io import wavfile

from echo.chirp import PeriodicPlayer, make_period
from echo.config import REPO_ROOT, Config, load_config, resolve_gain


def _sd() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise RuntimeError(
            "sounddevice/PortAudio is unavailable on this machine. Use --sim to run "
            "with the simulator instead."
        ) from exc
    return sd


LOOPBACK_HINTS = ("stereo mix", "loopback", "what u hear", "wave out")


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices() -> list[DeviceInfo]:
    sd = _sd()
    apis = sd.query_hostapis()
    return [
        DeviceInfo(
            index=i,
            name=d["name"],
            host_api=apis[d["hostapi"]]["name"],
            max_input_channels=int(d["max_input_channels"]),
            max_output_channels=int(d["max_output_channels"]),
            default_samplerate=float(d["default_samplerate"]),
        )
        for i, d in enumerate(sd.query_devices())
    ]


def format_devices(devices: list[DeviceInfo], preferred_api: str) -> str:
    rows = sorted(devices, key=lambda d: (d.host_api != preferred_api, d.host_api, d.index))
    lines = [f"{'id':>3}  {'in':>2} {'out':>3}  {'rate':>6}  {'host api':<22} name"]
    for d in rows:
        star = "*" if d.host_api == preferred_api else " "
        lines.append(
            f"{d.index:>3}{star} {d.max_input_channels:>2} {d.max_output_channels:>3}  "
            f"{d.default_samplerate:>6.0f}  {d.host_api:<22} {d.name}"
        )
    lines.append(f"(* = preferred host API '{preferred_api}')")
    return "\n".join(lines)


def resolve_device(
    spec: str | int | None, kind: str, preferred_api: str, devices: list[DeviceInfo] | None = None
) -> int:
    """Map a config device spec (None / index / name substring) to a device index.

    Name matches and defaults prefer devices on ``preferred_api`` (WASAPI), so
    input and output end up on the same host API, which duplex streams need.
    """
    if kind not in ("input", "output"):
        raise ValueError("kind must be 'input' or 'output'")
    devices = devices if devices is not None else list_devices()

    def usable(d: DeviceInfo) -> bool:
        return (d.max_input_channels if kind == "input" else d.max_output_channels) > 0

    if isinstance(spec, int):
        match = [d for d in devices if d.index == spec]
        if not match or not usable(match[0]):
            raise ValueError(f"Device {spec} is not a valid {kind} device.")
        return spec
    if isinstance(spec, str):
        hits = [d for d in devices if usable(d) and spec.lower() in d.name.lower()]
        if not hits:
            raise ValueError(f"No {kind} device matching {spec!r}. Run `python -m echo.audio_io list`.")
        hits.sort(key=lambda d: d.host_api != preferred_api)
        return hits[0].index
    # Default device of the preferred host API, else the global default.
    sd = _sd()
    for api in sd.query_hostapis():
        if api["name"] == preferred_api:
            idx = api["default_input_device" if kind == "input" else "default_output_device"]
            if idx is not None and idx >= 0:
                return int(idx)
    default = sd.default.device[0 if kind == "input" else 1]
    if default is None or default < 0:
        raise ValueError(f"No default {kind} device.")
    return int(default)


class DuplexSonar:
    """Full-duplex stream: one callback plays the chirp train and captures input.

    Playback and capture share a single callback and therefore a single sample
    clock, so the echo timing relative to the emitted chirps is fixed.

    Use as a context manager; iterate :meth:`blocks` for mono float32 input.
    """

    def __init__(self, cfg: Config, gain: float | None = None, allow_loud: bool = False,
                 keep_raw: bool = False) -> None:
        self.cfg = cfg
        self.allow_loud = allow_loud
        a = cfg.audio
        g = resolve_gain(cfg.chirp.gain if gain is None else gain, cfg.safety.max_gain, allow_loud)
        self._player = PeriodicPlayer(make_period(cfg.chirp, a.sample_rate), g)
        devices = list_devices()
        self.input_index = resolve_device(a.input_device, "input", a.host_api, devices)
        self.output_index = resolve_device(a.output_device, "output", a.host_api, devices)
        by_index = {d.index: d for d in devices}
        self.input_info, self.output_info = by_index[self.input_index], by_index[self.output_index]
        if any(k in self.input_info.name.lower() for k in LOOPBACK_HINTS):
            print(f"WARNING: input '{self.input_info.name}' looks like a loopback device; it records "
                  "the speaker signal digitally and will contain no echoes. Set audio.input_device "
                  "to your microphone.")
        if self.input_info.host_api != self.output_info.host_api:
            raise ValueError(
                f"Input ({self.input_info.host_api}) and output ({self.output_info.host_api}) "
                "must share a host API for a full-duplex stream."
            )
        self.in_channels = max(a.input_channels, a.input_channel + 1)
        if self.in_channels > self.input_info.max_input_channels:
            raise ValueError(f"Input device has only {self.input_info.max_input_channels} channel(s).")
        self.out_channels = min(a.output_channels, self.output_info.max_output_channels)
        self.device_name = f"in: {self.input_info.name} | out: {self.output_info.name}"
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._keep_raw = keep_raw
        self._raw_in: list[np.ndarray] = []
        self._raw_out: list[np.ndarray] = []
        self._mono = np.zeros(0, dtype=np.float32)
        self._stream: Any = None
        self.t0: float = 0.0  # time.monotonic() of the first captured sample
        self.xruns = 0
        self.dropped_blocks = 0
        self.status_messages: list[str] = []

    # -- gain -----------------------------------------------------------------
    @property
    def gain(self) -> float:
        return self._player.gain

    def set_gain(self, gain: float) -> None:
        self._player.gain = resolve_gain(gain, self.cfg.safety.max_gain, self.allow_loud)

    # -- stream ---------------------------------------------------------------
    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int,
                  time_info: Any, status: Any) -> None:
        if self.t0 == 0.0:
            # Map the ADC time of this buffer onto time.monotonic().
            lag = max(0.0, float(time_info.currentTime - time_info.inputBufferAdcTime))
            self.t0 = time.monotonic() - lag
        if status:
            self.xruns += 1
            if len(self.status_messages) < 50:
                self.status_messages.append(str(status))
        if len(self._mono) != frames:
            self._mono = np.zeros(frames, dtype=np.float32)
        self._player.fill_into(self._mono)
        outdata[:] = self._mono[:, None]
        if self._keep_raw:
            self._raw_out.append(self._mono.copy())
        try:
            self._q.put_nowait(indata[:, self.cfg.audio.input_channel].copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self) -> None:
        sd = _sd()
        a = self.cfg.audio
        extra = None
        if self.input_info.host_api == "Windows WASAPI":
            ws = sd.WasapiSettings(exclusive=a.wasapi_exclusive, auto_convert=not a.wasapi_exclusive)
            extra = (ws, ws)
        self._stream = sd.Stream(
            samplerate=a.sample_rate,
            device=(self.input_index, self.output_index),
            channels=(self.in_channels, self.out_channels),
            dtype="float32",
            latency=a.latency,
            extra_settings=extra,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> DuplexSonar:
        self.start()
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.stop()

    def blocks(self, timeout_s: float = 2.0) -> Iterator[np.ndarray]:
        """Yield captured mono blocks until the stream stops."""
        while self._stream is not None:
            try:
                b = self._q.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise RuntimeError("No audio received for 2 s; is the microphone enabled?") from exc
            if self._keep_raw:
                self._raw_in.append(b)
            yield b

    @property
    def raw_capture(self) -> np.ndarray:
        return np.concatenate(self._raw_in) if self._raw_in else np.zeros(0, np.float32)

    @property
    def raw_playback(self) -> np.ndarray:
        return np.concatenate(self._raw_out) if self._raw_out else np.zeros(0, np.float32)


def save_wavs(out_dir: Path, sample_rate: int, capture: np.ndarray, playback: np.ndarray | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_dir / "capture.wav", sample_rate, capture.astype(np.float32))
    if playback is not None:
        wavfile.write(out_dir / "playback.wav", sample_rate, playback.astype(np.float32))
    return out_dir


def debug_dir() -> Path:
    return REPO_ROOT / "debug" / datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list audio devices")
    run = sub.add_parser("run", help="duplex run that saves raw WAVs")
    run.add_argument("--seconds", type=float, default=10.0)
    run.add_argument("--gain", type=float, default=None)
    run.add_argument("--allow-loud", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "list":
        print(format_devices(list_devices(), cfg.audio.host_api))
        return
    sonar = DuplexSonar(cfg, gain=args.gain, allow_loud=args.allow_loud, keep_raw=True)
    print(f"Duplex: {sonar.device_name} at gain {sonar.gain:.3f}")
    need = int(args.seconds * cfg.audio.sample_rate)
    got = 0
    with sonar:
        for b in sonar.blocks():
            got += len(b)
            if got >= need:
                break
    cap = sonar.raw_capture
    out = save_wavs(debug_dir(), cfg.audio.sample_rate, cap, sonar.raw_playback)
    print(f"Saved {len(cap) / cfg.audio.sample_rate:.1f} s to {out} (xruns: {sonar.xruns})")
    print("For timing and jitter use: python -m echo.diagnostics timing")


if __name__ == "__main__":
    main()
