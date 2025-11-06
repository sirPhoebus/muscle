from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    segments: int = 6                # genetic segment count (3..8 recommended)
    wave_freq: float = 1.2           # Hz
    wave_amp: float = 0.55           # 0..1 amplitude
    steer_gain: float = 1.3          # sensor delta to muscle bias
    forward_gain: float = 28.0       # baseline speed scale
    sensor_fov_deg: float = 35.0     # degrees for left/right sensors
    sensor_range: float = 160.0
    avoidance_threshold: float = 90.0  # legacy head-prox threshold (kept as backup)
    sensor_end_contact_px: float = 30.0  # how close to the antenna END triggers avoidance
    avoid_turn_gain: float = 2.0       # extra steering when antenna end nears wall
    avoid_slow_gain: float = 1.2       # slow forward speed as end gets close
    sensor_fan_rays: int = 7          # number of rays over FOV (odd preferred)
    avoid_force_gain: float = 2.2      # weight for vector-field repulsion
    min_forward_frac: float = 0.15     # minimum fraction of forward speed
    k_gain_head: float = 1.2
    k_gain_bg: float = 1.0
    head_interval_ms: int = 260
    bg_interval_ms: int = 340
    muscle_baseline: float = 0.05
    muscle_gain: float = 2.2
    muscle_gamma: float = 0.8
    neurons_enabled: bool = True
    attract_gain: float = 1.6        # food attraction strength
    food_range: float = 420.0        # range within which food attracts
    brain_gain: float = 1.5          # brain steering strength
    brain_interval_ms: int = 160     # twitch cadence
    brain_twitch_gain: float = 1.3   # scales brain twitch amplitude
    brain_stuck_timeout_s: float = 6.0       # time before declaring target stuck
    brain_stuck_min_improve_px: float = 24.0 # required improvement to keep target
    brain_target_cooldown_s: float = 10.0    # ignore stuck target for this long
    joint_flex: float = 1.0          # >1 more bend and looser joints; <1 stiffer


def create_random_genetics(base_params: Params | None = None, variation: float = 0.3, rng=None) -> Params:
    """Create a new Params with random genetics across full ranges."""
    import random as _random
    import copy

    if rng is None:
        rng = _random

    if base_params is None:
        base_params = Params()

    # Create a copy
    params = copy.deepcopy(base_params)

    # Helper to randomize uniformly across the full range
    def vary_float(value: float, min_val: float, max_val: float) -> float:
        return float(rng.uniform(min_val, max_val))

    def vary_int(value: int, min_val: int, max_val: int) -> int:
        # randint is inclusive on both ends
        return int(rng.randint(int(min_val), int(max_val)))

    # Vary each genetic parameter
    params.segments = vary_int(getattr(params, 'segments', 6), 3, 8)
    params.wave_freq = vary_float(params.wave_freq, 0.3, 3.0)
    params.wave_amp = vary_float(params.wave_amp, 0.1, 1.0)
    params.steer_gain = vary_float(params.steer_gain, 0.2, 3.0)
    params.forward_gain = vary_float(params.forward_gain, 10.0, 60.0)
    params.sensor_fov_deg = vary_float(params.sensor_fov_deg, 10.0, 80.0)
    params.sensor_range = vary_float(params.sensor_range, 60.0, 250.0)
    params.avoid_turn_gain = vary_float(params.avoid_turn_gain, 0.5, 4.0)
    params.avoid_slow_gain = vary_float(params.avoid_slow_gain, 0.5, 2.5)
    params.sensor_fan_rays = vary_int(params.sensor_fan_rays, 3, 13)
    params.avoid_force_gain = vary_float(params.avoid_force_gain, 0.5, 4.0)
    params.min_forward_frac = vary_float(params.min_forward_frac, 0.05, 0.4)
    params.k_gain_head = vary_float(params.k_gain_head, 0.5, 2.5)
    params.k_gain_bg = vary_float(params.k_gain_bg, 0.5, 2.5)
    params.head_interval_ms = vary_int(params.head_interval_ms, 100, 600)
    params.bg_interval_ms = vary_int(params.bg_interval_ms, 150, 800)
    params.muscle_baseline = vary_float(params.muscle_baseline, 0.01, 0.2)
    params.muscle_gain = vary_float(params.muscle_gain, 1.0, 4.0)
    params.muscle_gamma = vary_float(params.muscle_gamma, 0.5, 2.0)
    params.attract_gain = vary_float(params.attract_gain, 0.5, 3.0)
    params.food_range = vary_float(params.food_range, 200.0, 600.0)
    params.brain_gain = vary_float(params.brain_gain, 0.5, 3.0)
    params.brain_interval_ms = vary_int(params.brain_interval_ms, 80, 300)
    params.brain_twitch_gain = vary_float(params.brain_twitch_gain, 0.5, 2.5)
    params.joint_flex = vary_float(params.joint_flex, 0.4, 2.0)

    return params
