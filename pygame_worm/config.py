from __future__ import annotations

from typing import List, Tuple
import colorsys

# ----- Tunables -----
WIN_SIZE: Tuple[int, int] = (1600, 800)
WORLD_SCALE: int = 10
WORLD_SIZE: Tuple[int, int] = (WIN_SIZE[0] * WORLD_SCALE, WIN_SIZE[1] * WORLD_SCALE)
TARGET_FPS: int = 60

# Base body parameters (will be varied per worm)
BASE_SEGMENTS: int = 18
BASE_SEG_SPACING: float = 16.0
BASE_BODY_RADIUS: float = 10.0

# Base undulation (central pattern) for locomotion
WAVE_PHASE_PER_SEG: float = 0.45  # radians per segment

# Life simulation globals
STARVATION_TIME_S: float = 300.0           # die if no food within this time (~5 minutes)
REPRODUCTION_FOOD_REQUIRED: int = 5        # foods needed to be "ready"
REPRODUCTION_DISTANCE_SCALE: float = 1.0   # how close heads must be (~ sum of radii)
REPRODUCTION_COOLDOWN_S: float = 5.0       # seconds before a parent can reproduce again

# Food spawn timing
FOOD_SPAWN_MIN_S: float = 2.0
FOOD_SPAWN_MAX_S: float = 12.0
FOOD_TARGET_PER_WORM: float = 1.2  # aim for this many available foods per worm (more initial food)
FOOD_SPAWN_BURST_MAX: int = 10     # max foods to add in one spawn
FOOD_LOW_RATIO: float = 0.30       # if available/total <= this, boost spawn
FOOD_LOW_RATIO_MULTIPLIER: int = 3 # multiplier when ratio is low

# DNA visualization parameter specs: (attr_name, min, max)
DNA_SPECS: List[Tuple[str, float, float]] = [
    ("wave_freq", 0.3, 3.0),
    ("wave_amp", 0.1, 1.0),
    ("forward_gain", 10.0, 60.0),
    ("steer_gain", 0.2, 3.0),
    ("sensor_range", 60.0, 250.0),
    ("avoid_turn_gain", 0.5, 4.0),
    ("avoid_slow_gain", 0.5, 2.5),
    ("sensor_fan_rays", 3.0, 13.0),
    ("avoid_force_gain", 0.5, 4.0),
    ("min_forward_frac", 0.05, 0.4),
    ("k_gain_head", 0.5, 2.5),
    ("k_gain_bg", 0.5, 2.5),
    ("head_interval_ms", 100.0, 600.0),
    ("bg_interval_ms", 150.0, 800.0),
    ("muscle_baseline", 0.01, 0.2),
    ("muscle_gain", 1.0, 4.0),
    ("muscle_gamma", 0.5, 2.0),
    ("attract_gain", 0.5, 3.0),
    ("food_range", 200.0, 600.0),
    ("brain_gain", 0.5, 3.0),
    ("brain_interval_ms", 80.0, 300.0),
    ("brain_twitch_gain", 0.5, 2.5),
    ("joint_flex", 0.4, 2.0),
]


def _make_dna_colors(n: int) -> List[Tuple[int, int, int]]:
    cols: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = (i / max(1, n))  # 0..1
        s = 0.65
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


DNA_COLORS: List[Tuple[int, int, int]] = _make_dna_colors(len(DNA_SPECS))
