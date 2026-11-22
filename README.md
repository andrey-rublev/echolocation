# echolocation

Laptop-only sonar: chirps from the speaker, echoes into the mic, a small model
infers presence, distance and room identity.

> **Warning:** do not run this loud near ears or small speakers. Gain starts at
> 0.1 of full scale and is capped by `safety.max_gain`.

Setup (Windows, native Python, not WSL):

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe numpy scipy sounddevice matplotlib pyyaml pytest opencv-python mediapipe
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pytest
```

The full guide is written in Stage 5.
