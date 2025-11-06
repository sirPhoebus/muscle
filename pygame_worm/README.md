Pygame Worm — Neuron + Muscle Fibers

Overview
- Segmented worm simulation with neuron-triggered muscle twitches and undulatory gait.
- Sensors and an optional brain guide movement to avoid obstacles and collect food.
- Supports large populations with CPU optimizations and an optional GPU renderer (ModernGL).

Install
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r pygame_worm/requirements.txt
```

Run
```bash
# CPU (default)
python -m pygame_worm.main

# GPU rendering (ModernGL)
python -m pygame_worm.main --gl
# or
GL=1 python -m pygame_worm.main
```

Controls
- R: regenerate maze
- SPACE: toggle pause
- ESC / close window: quit
- Click “Neurons (Main Worm)” (UI panel) to toggle neural drive.

Stats & Dashboard
- On exit, a CSV is written to `./stats/stats_<timestamp>.csv`.
- Generate an HTML dashboard for the latest CSV:
```bash
python tools/make_dashboard.py
# Output: stats/dashboard_latest.html
```
(The dashboard uses Chart.js via CDN; no extra Python deps.)

Project Structure
- `config.py` — constants and DNA color palette
- `params.py` — genetic parameters + `create_random_genetics()`
- `worm.py` — worm entity (update/draw), muscles, sensors, brain
- `maze.py` — world, obstacles, foods, drawing, spatial queries
- `spatial.py` — 2D spatial hash for proximity queries
- `ui.py` — sliders panel and button
- `gl_renderer.py` — GPU instanced renderer (ModernGL)
- `app.py` — orchestration (loop, camera, spawn, stats)
- `main.py` — thin entry wrapper
- `tools/make_dashboard.py` — build HTML dashboard from latest stats CSV
