"""`python -m echo` prints the workflow; every step is its own module."""

COMMANDS = """\
echolocation - laptop sonar. Every command accepts --help; live ones accept --sim.

  1. audio check     python -m echo.audio_io list
                     python -m echo.diagnostics timing --seconds 10 --save-wav
                     python -m echo.diagnostics linearity
  2. calibrate       python -m echo.calibrate --distance 0.50
     waterfall       python -m echo.live_waterfall
  3. collect         python -m echo.collect --room dorm_desk --label empty --minutes 1
                     python -m echo.camera download-model
                     python -m echo.camera calibrate --distance 1.5
                     python -m echo.collect --room dorm_desk --label auto --camera --minutes 3
                     python -m echo.viz sessions <a.npz> <b.npz> --out docs/sessions.png
  4. train / eval    python -m echo.train --name first_run
                     python -m echo.evaluate --name first_run
  5. live demo       python -m echo.live_infer --name first_run

  no hardware?       python -m echo.sim_dataset --out data/sim_sessions
                     python -m echo.live_waterfall --sim --sim-person walk
"""

if __name__ == "__main__":
    print(COMMANDS)
