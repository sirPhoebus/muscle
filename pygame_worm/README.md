Pygame Worm — Neuron + Muscle Fibers

Overview
- Reproduces the essentials from the `muscle-fiber` PoC: neurons triggering twitches that drive muscle fibers with a rise/hold/decay envelope mapped to contraction.
- Simulates a segmented worm in Pygame with left/right longitudinal muscle bands per segment.
- Adds two sensory neurons/rays on the head that detect obstacles (maze walls) and steer the worm left/right to avoid collisions.

What you get
- `Neuron` produces stochastic twitch amplitudes via `sigmoid(k * N(0,1))`.
- `Twitch` envelope: rise → hold → decay.
- `MuscleFiber` maps twitch activation to a contraction value.
- `Worm` with segments and two fiber bands (left/right) per segment; undulatory gait with steering bias from sensors.
- `Maze` with simple rectangular obstacles.

Requirements
- Python 3.9+
- Pygame 2.x
- NumPy

Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r pygame_worm/requirements.txt
```

Run
```bash
# Preferred
python -m pygame_worm.main

# Or (fallback):
python pygame_worm/main.py
```

Controls
- `R`: regenerate maze
- `SPACE`: toggle pause
- `ESC`/window close: quit
- Click the `Neurons` button (top of panel) to toggle automatic neuron firing on/off.

Notes
- Rendering is 2.5D (top-down) for simplicity within Pygame. The muscle model and control logic reflect the neuron→fiber twitch concepts from the TS PoC.
- Key parameters (segment count, frequencies, gains) are at the top of `main.py`.
