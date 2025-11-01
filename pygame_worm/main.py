from __future__ import annotations
import sys
import math
from dataclasses import dataclass
import json
from typing import Callable, List, Tuple

import pygame as pg
import numpy as np
import colorsys

# Support running as a script or module
try:
    from .neuro import Neuron
    from .muscle import MuscleFiber
    from .maze import Maze, Food
    from .neuro import Twitch, compute_twitch_timing
    from .brain import Brain
except ImportError:  # pragma: no cover - fallback for direct script execution
    from neuro import Neuron
    from muscle import MuscleFiber
    from maze import Maze, Food
    from neuro import Twitch, compute_twitch_timing
    from brain import Brain


# ----- Tunables -----
WIN_SIZE = (900, 900)
WORLD_SCALE = 10
WORLD_SIZE = (WIN_SIZE[0] * WORLD_SCALE, WIN_SIZE[1] * WORLD_SCALE)
TARGET_FPS = 60

# Base body parameters (will be varied per worm)
BASE_SEGMENTS = 18
BASE_SEG_SPACING = 16.0
BASE_BODY_RADIUS = 10.0

# Base undulation (central pattern) for locomotion
WAVE_PHASE_PER_SEG = 0.45  # radians per segment

# Life simulation globals
STARVATION_TIME_S = 60.0            # die if no food within this time
REPRODUCTION_FOOD_REQUIRED = 5      # foods needed to be "ready"
REPRODUCTION_DISTANCE_SCALE = 1.0   # how close heads must be (× sum of radii)
REPRODUCTION_COOLDOWN_S = 5.0       # seconds before a parent can reproduce again



# Food spawn timing
FOOD_SPAWN_MIN_S = 2.0
FOOD_SPAWN_MAX_S = 12.0
FOOD_TARGET_PER_WORM = 0.6  # aim for this many available foods per worm
FOOD_SPAWN_BURST_MAX = 10   # max foods to add in one spawn

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

# Precompute a distinct color per DNA bar using evenly spaced HSV hues
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


@dataclass
class Params:
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


@dataclass
class Segment:
    pos: np.ndarray  # shape (2,)
    dir: float       # heading angle radians
    left: MuscleFiber
    right: MuscleFiber


class Worm:
    def __init__(self, origin: Tuple[float, float], heading: float, params: Params, color: Tuple[int, int, int] | None = None, 
                 segments: int | None = None, body_radius: float | None = None, seg_spacing: float | None = None):
        # Each worm has its own genetic parameters
        self.params = params
        
        # Physical characteristics based on genetics
        self.num_segments = segments if segments is not None else BASE_SEGMENTS
        self.body_radius = body_radius if body_radius is not None else BASE_BODY_RADIUS
        self.seg_spacing = seg_spacing if seg_spacing is not None else BASE_SEG_SPACING
        
        self.segs: List[Segment] = []
        x, y = origin
        for i in range(self.num_segments):
            pos = np.array([x - i * self.seg_spacing * math.cos(heading), y - i * self.seg_spacing * math.sin(heading)], dtype=float)
            self.segs.append(Segment(pos=pos, dir=heading, left=MuscleFiber(), right=MuscleFiber()))
        
        # One neuron per fiber band on the head (optional); plus global background neuron
        self.head_neuron_L = Neuron(interval_ms=params.head_interval_ms, k_gain=params.k_gain_head)
        self.head_neuron_R = Neuron(interval_ms=params.head_interval_ms, k_gain=params.k_gain_head)
        self.bg_neuron = Neuron(interval_ms=params.bg_interval_ms, k_gain=params.k_gain_bg)
        self.brain = Brain(interval_ms=params.brain_interval_ms)
        self.paused = False
        self.color = color if color is not None else (230, 232, 245)
        # Random phase offset so worms don't all wiggle in sync
        self.phase_offset = np.random.uniform(0, 2 * math.pi)
        # Cache last sensor distances (for HUD/drawing)
        self.last_sensor_L: float = params.sensor_range
        self.last_sensor_R: float = params.sensor_range
        self.last_sensor_origin_L: Tuple[float, float] = origin
        self.last_sensor_origin_R: Tuple[float, float] = origin
        self.last_sensor_fan: List[Tuple[Tuple[float, float], float, float]] = []  # (origin, angle, dist)
        self.last_food_target: Tuple[float, float] | None = None
        # Targeting and avoidance of unreachable foods
        self.current_target: Food | None = None
        self.target_start_t: float = 0.0
        self.target_start_dist: float = float('inf')
        self.target_best_dist: float = float('inf')
        self.avoid_food_until: dict[int, float] = {}
        # Apply muscle params using own genetics
        self.apply_muscle_params(self.params)
        # Life sim state
        self.alive: bool = True
        self.food_eaten: int = 0
        self.time_since_last_eat: float = 0.0
        self.ready_to_mate: bool = False
        self.last_repro_time: float = -1e6
        # Speed meter
        self.current_speed: float = 0.0
        self.speed_ema: float = 0.0
        self.max_speed: float = 0.0

    def dna_values(self) -> List[float]:
        # Normalize configured genetic parameters 0..1 for DNA bars
        vals: List[float] = []
        p = self.params
        for name, mn, mx in DNA_SPECS:
            v = getattr(p, name)
            # Cast ints to float for normalization
            vf = float(v)
            if mx <= mn:
                n = 0.0
            else:
                n = (vf - mn) / (mx - mn)
            # clamp
            n = 0.0 if n < 0.0 else (1.0 if n > 1.0 else n)
            vals.append(n)
        return vals

    def draw_dna_tag(self, surf: pg.Surface, cam: Tuple[float, float]) -> None:
        # Visual DNA tag above head: multicolored bars, heights 0..1
        hx, hy = self.head_pos()
        cx, cy = cam
        x = int(hx - cx)
        y = int(hy - cy)
        bars = self.dna_values()
        if not bars:
            return
        # Layout
        bar_w = 4
        gap = 2
        padding = 6
        max_h = 28
        total_w = len(bars) * bar_w + (len(bars) - 1) * gap + 2 * padding
        total_h = max_h + 2 * padding
        # Position tag centered above head
        tag_x = x - total_w // 2
        tag_y = y - int(self.body_radius) - total_h - 6
        rect = pg.Rect(tag_x, tag_y, total_w, total_h)
        # Save for hit-testing (screen space)
        self.last_tag_rect = rect.copy()
        # Background and border
        bg = (18, 22, 32)
        bd = (70, 80, 110) if not self.ready_to_mate else (70, 160, 90)
        radius = 8
        pg.draw.rect(surf, bg, rect, border_radius=radius)
        pg.draw.rect(surf, bd, rect, width=1, border_radius=radius)
        # Draw bars bottom-aligned inside rect
        bx = tag_x + padding
        by = tag_y + padding
        for i, n in enumerate(bars):
            h = int(max(1, n * max_h))
            col = DNA_COLORS[i % len(DNA_COLORS)]
            bx_i = bx + i * (bar_w + gap)
            by_i = by + (max_h - h)
            pg.draw.rect(surf, col, pg.Rect(bx_i, by_i, bar_w, h), border_radius=2)
        # Food eaten counter (small text) and speed meter
        try:
            if not hasattr(Worm, "_dna_font"):
                Worm._dna_font = pg.font.SysFont(None, 12)
            font = Worm._dna_font
            txt = font.render(f"F:{self.food_eaten}", True, (220, 230, 240))
            surf.blit(txt, (tag_x + 6, tag_y + 4))
            spd = font.render(f"S:{int(self.speed_ema):d}", True, (210, 220, 255))
            surf.blit(spd, (tag_x + rect.w - 36, tag_y + 4))
        except Exception:
            pass

    def apply_muscle_params(self, params: Params) -> None:
        for seg in self.segs:
            seg.left.baseline = params.muscle_baseline
            seg.left.gain = params.muscle_gain
            seg.left.gamma = params.muscle_gamma
            seg.right.baseline = params.muscle_baseline
            seg.right.gain = params.muscle_gain
            seg.right.gamma = params.muscle_gamma

    def head_pos(self) -> Tuple[float, float]:
        p = self.segs[0].pos
        return float(p[0]), float(p[1])

    def head_dir(self) -> float:
        return self.segs[0].dir

    def update(self, dt_ms: int, t_sec: float, maze: Maze) -> None:
        params = self.params  # Use worm's own genetic parameters
        if not self.alive:
            return
        if self.paused:
            return
        dt = dt_ms / 1000.0
        # Starvation timer
        self.time_since_last_eat += dt
        if self.time_since_last_eat >= STARVATION_TIME_S:
            self.alive = False
            return

        # Sensor rays from head (throttled)
        hx, hy = self.head_pos()
        hdir = self.head_dir()
        sensor_fov = math.radians(params.sensor_fov_deg)
        # Determine if we should recompute sensors this frame
        # Throttle based on worm id to stagger updates
        if not hasattr(self, "_sensor_frame_mod"):
            self._sensor_frame_mod = np.random.randint(0, 3)
        recompute = (int(t_sec * 60) + self._sensor_frame_mod) % 3 == 0
        if recompute:
            # Build a fan of rays across [-FOV, +FOV]
            n = max(3, int(params.sensor_fan_rays))
            # Dynamic cap to reduce cost with many agents
            n = int(max(3, min(n, 9)))
            if n % 2 == 0:
                n += 1  # prefer odd to have a center ray
            fan: List[Tuple[Tuple[float, float], float, float]] = []
            for i in range(n):
                t = -1.0 + 2.0 * (i / max(1, n - 1))  # -1..1
                ang = hdir + t * sensor_fov
                tip = (hx + self.body_radius * math.cos(ang), hy + self.body_radius * math.sin(ang))
                d = maze.raycast(tip, ang, params.sensor_range)
                fan.append((tip, ang, d))
            self.last_sensor_fan = fan
            # Derive endpoints
            angL = hdir - sensor_fov
            angR = hdir + sensor_fov
            tipL = (hx + self.body_radius * math.cos(angL), hy + self.body_radius * math.sin(angL))
            tipR = (hx + self.body_radius * math.cos(angR), hy + self.body_radius * math.sin(angR))
            dL = maze.raycast(tipL, angL, params.sensor_range)
            dR = maze.raycast(tipR, angR, params.sensor_range)
            tipC = (hx + self.body_radius * math.cos(hdir), hy + self.body_radius * math.sin(hdir))
            dC = maze.raycast(tipC, hdir, params.sensor_range)
            self._last_dists = (dL, dR, dC)
            # cache for drawing
            self.last_sensor_L = dL
            self.last_sensor_R = dR
            self.last_sensor_origin_L = tipL
            self.last_sensor_origin_R = tipR
        else:
            # Reuse last results if available
            dL, dR, dC = getattr(self, "_last_dists", (params.sensor_range, params.sensor_range, params.sensor_range))
            self.last_sensor_L = dL
            self.last_sensor_R = dR
        # Sensor bias: steer away from nearer obstacle
        bias = 0.0
        # Vector-field avoidance using the multi-ray fan
        win = max(1.0, params.sensor_end_contact_px)
        rep_x = rep_y = 0.0
        cEndC = max(0.0, min(1.0, (win - dC) / win))
        for (tip, ang, dist) in self.last_sensor_fan:
            c = max(0.0, min(1.0, (win - dist) / win))
            if c <= 0.0:
                continue
            w = (c * c)
            rep_x += -w * math.cos(ang)
            rep_y += -w * math.sin(ang)
        lat_x = -math.sin(hdir)
        lat_y = math.cos(hdir)
        rep_lat = rep_x * lat_x + rep_y * lat_y
        bias += params.avoid_turn_gain * rep_lat
        # Legacy near-head avoidance as a mild backup when very close
        if min(dL, dR) < params.avoidance_threshold:
            bias += 0.5 * params.steer_gain * (dR - dL) / max(1.0, params.sensor_range)
        # When neurons are disabled, there is no neural drive to effect steering
        if not params.neurons_enabled:
            bias = 0.0

        # Food attraction: steer toward resultant vector of nearby foods (spatial query)
        if params.neurons_enabled and params.attract_gain > 0.0:
            hx, hy = self.head_pos()
            acc = np.array([0.0, 0.0], dtype=float)
            for f in maze.foods_near(hx, hy, params.food_range):
                dx = f.x - hx
                dy = f.y - hy
                dist = math.hypot(dx, dy)
                if dist > 1e-3:
                    w = max(0.0, 1.0 - dist / params.food_range)
                    acc[0] += (w / dist) * dx
                    acc[1] += (w / dist) * dy
            mag = float(np.linalg.norm(acc))
            if mag > 1e-6:
                acc /= mag
                target_ang = math.atan2(acc[1], acc[0])
                # smallest signed angle difference
                ang = target_ang - hdir
                while ang > math.pi:
                    ang -= 2 * math.pi
                while ang < -math.pi:
                    ang += 2 * math.pi
                # scale by gain and attraction magnitude (cap at 1)
                bias += params.attract_gain * (ang / math.pi) * min(1.0, mag)

        # Brain: integrate sensors and exact food vector for steering and twitches
        if params.neurons_enabled and params.brain_gain > 0.0:
            # Determine target food, prefer existing unless eaten/invalid or on cooldown
            hx, hy = self.head_pos()
            now = t_sec
            # Helper to check if a food is avoid-listed
            def _is_avoided(f: Food) -> bool:
                until = self.avoid_food_until.get(id(f))
                return until is not None and now < until

            # Validate current target
            if self.current_target is None or self.current_target.eaten or _is_avoided(self.current_target):
                self.current_target = None
            # Select nearest non-avoided food if no current target
            if self.current_target is None:
                best_f: Food | None = None
                best_d = float('inf')
                for f in maze.foods_near(hx, hy, params.food_range):
                    if _is_avoided(f):
                        continue
                    d = math.hypot(f.x - hx, f.y - hy)
                    if d < best_d:
                        best_d = d
                        best_f = f
                if best_f is not None:
                    self.current_target = best_f
                    self.target_start_t = now
                    self.target_start_dist = best_d
                    self.target_best_dist = best_d

            # Compute angle and distance to current target
            nearest_ang = 0.0
            nearest_dist = float('inf')
            nearest_xy = None
            if self.current_target is not None:
                dx = self.current_target.x - hx
                dy = self.current_target.y - hy
                nearest_dist = math.hypot(dx, dy)
                nearest_ang = math.atan2(dy, dx) - hdir
                nearest_xy = (self.current_target.x, self.current_target.y)
                # Track progress and detect stuck
                if nearest_dist < self.target_best_dist:
                    self.target_best_dist = nearest_dist
                elapsed = now - self.target_start_t
                improved = self.target_start_dist - self.target_best_dist
                if (elapsed >= params.brain_stuck_timeout_s and
                    improved < params.brain_stuck_min_improve_px and
                    nearest_dist > self.body_radius * 1.3):
                    # Mark as avoided for cooldown and drop target
                    self.avoid_food_until[id(self.current_target)] = now + params.brain_target_cooldown_s
                    self.current_target = None

            while nearest_ang > math.pi:
                nearest_ang -= 2 * math.pi
            while nearest_ang < -math.pi:
                nearest_ang += 2 * math.pi
            self.last_food_target = nearest_xy if nearest_xy is not None else None

            out = self.brain.update(
                dt_ms,
                dL,
                dR,
                nearest_ang,
                nearest_dist if nearest_dist != float('inf') else params.food_range,
                params.sensor_range,
                params.food_range,
            )
            bias += params.brain_gain * out.steer
            # Convert brain twitches to head fiber twitches
            if out.twitch_left > 0.0:
                amp = max(0.0, min(1.0, out.twitch_left * params.brain_twitch_gain))
                r, h, d = compute_twitch_timing(amp)
                self.segs[0].left.trigger(Twitch(target=amp, rise=r, hold=h, decay=d))
            if out.twitch_right > 0.0:
                amp = max(0.0, min(1.0, out.twitch_right * params.brain_twitch_gain))
                r, h, d = compute_twitch_timing(amp)
                self.segs[0].right.trigger(Twitch(target=amp, rise=r, hold=h, decay=d))

        if params.neurons_enabled:
            # Background neuron occasionally creates twitch on random mid-body segment
            tw = self.bg_neuron.update(dt_ms)
            if tw is not None:
                idx = np.random.randint(2, self.num_segments)
                # pick left or right randomly
                if np.random.rand() < 0.5:
                    self.segs[idx].left.trigger(tw)
                else:
                    self.segs[idx].right.trigger(tw)

            # Head-specific neurons modulate L/R for more variety
            twL = self.head_neuron_L.update(dt_ms)
            if twL is not None:
                self.segs[0].left.trigger(twL)
            twR = self.head_neuron_R.update(dt_ms)
            if twR is not None:
                self.segs[0].right.trigger(twR)
        else:
            # When disabled, clear active twitches and zero activations
            for seg in self.segs:
                seg.left.twitch = None
                seg.right.twitch = None
                seg.left.activation = 0.0
                seg.right.activation = 0.0

        # Drive undulatory gait + steering
        omega = 2.0 * math.pi * params.wave_freq
        for i, seg in enumerate(self.segs):
            phase = omega * t_sec + i * WAVE_PHASE_PER_SEG + self.phase_offset
            # When neurons are disabled, suppress the central pattern drive
            # Reduce gait amplitude near obstacles to help turning
            # Use maximum closeness over the sensor fan (including center)
            avoid_strength = cEndC
            for (_, _, dist) in self.last_sensor_fan:
                avoid_strength = max(avoid_strength, max(0.0, min(1.0, (max(1.0, params.sensor_end_contact_px) - dist) / max(1.0, params.sensor_end_contact_px))))
            amp_scale = 1.0 - 0.5 * avoid_strength
            base = ((math.sin(phase) * params.wave_amp) * amp_scale) if params.neurons_enabled else 0.0
            # Steering bias: positive bias increases left activation (turn left)
            if params.neurons_enabled:
                seg.left.update()
                seg.right.update()
            left_a = np.clip(base + bias + seg.left.activation, 0.0, 1.0)
            right_a = np.clip(-base - bias + seg.right.activation, 0.0, 1.0)
            # Convert L/R activation delta to curvature; scaled by joint flex
            turn = (left_a - right_a) * 0.25 * max(0.1, params.joint_flex)
            seg.dir += turn * dt

        # Kinematics: head moves forward; each next segment follows previous at fixed spacing
        # Forward speed from head undulation; halt when neurons are disabled
        head_speed = 0.0 if not params.neurons_enabled else (
            params.forward_gain * (0.35 + 0.65 * abs(math.sin(omega * t_sec + self.phase_offset)))
        )
        # Slow down as antenna ends approach obstacles to avoid collisions; do not fully stop
        if params.neurons_enabled:
            avoid_strength = cEndC
            for (_, _, dist) in self.last_sensor_fan:
                avoid_strength = max(avoid_strength, max(0.0, min(1.0, (max(1.0, params.sensor_end_contact_px) - dist) / max(1.0, params.sensor_end_contact_px))))
            slow = max(0.0, 1.0 - params.avoid_slow_gain * avoid_strength)
            head_speed *= max(params.min_forward_frac, slow)
        # Update speedmeter
        self.current_speed = float(head_speed)
        self.speed_ema = 0.85 * self.speed_ema + 0.15 * self.current_speed
        if self.current_speed > self.max_speed:
            self.max_speed = self.current_speed
        vx = head_speed * math.cos(self.segs[0].dir)
        vy = head_speed * math.sin(self.segs[0].dir)
        self.segs[0].pos += np.array([vx * dt, vy * dt])

        # Obstacle/border response: make a 60° turn away and extricate slightly
        if maze.collide_circle(self.head_pos(), self.body_radius):
            # Choose turn direction based on which antenna side is freer
            turn_sign = 1.0 if dL > dR else -1.0
            self.segs[0].dir += turn_sign * math.radians(60.0)
            ndir = self.segs[0].dir
            # small extrication step along new heading to get out of contact
            step_out = self.body_radius * 0.9
            self.segs[0].pos += np.array([math.cos(ndir) * step_out, math.sin(ndir) * step_out])

        # Follow-the-leader for the rest
        for i in range(1, self.num_segments):
            prev = self.segs[i - 1]
            cur = self.segs[i]
            delta = prev.pos - cur.pos
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                continue
            dir_vec = delta / dist
            target = prev.pos - dir_vec * self.seg_spacing
            # springlike follow; higher joint_flex lowers stiffness to allow more bend
            follow_gain = 10.0 / max(0.2, params.joint_flex)
            cur.pos += (target - cur.pos) * min(1.0, follow_gain * dt)
            cur.dir = math.atan2(prev.pos[1] - cur.pos[1], prev.pos[0] - cur.pos[0])

        # Eat food: consume items very near head
        hx, hy = self.head_pos()
        eat_r = self.body_radius * 1.2
        er2 = eat_r * eat_r
        for f in maze.foods_near(hx, hy, eat_r):
            dx = f.x - hx
            dy = f.y - hy
            if (dx * dx + dy * dy) <= er2 and not f.eaten:
                maze.mark_food_eaten(f)
                self.food_eaten += 1
                self.time_since_last_eat = 0.0
                if self.food_eaten >= REPRODUCTION_FOOD_REQUIRED:
                    self.ready_to_mate = True

    def draw(self, surf: pg.Surface, cam: Tuple[float, float], draw_sensors: bool = False, draw_dna: bool = False, lite_mode: bool = False) -> None:
        params = self.params  # Use worm's own genetic parameters
        if not self.alive:
            return
        # Draw sensors (only for main worm)
        if draw_sensors:
            hx, hy = self.head_pos()
            hdir = self.head_dir()
            sensor_fov = math.radians(params.sensor_fov_deg)
            cx, cy = cam
            # draw sensor fan rays up to measured distances
            col = (160, 190, 255)
            for (origin, ang, dist) in getattr(self, 'last_sensor_fan', []):
                ox, oy = origin
                ex = ox + dist * math.cos(ang)
                ey = oy + dist * math.sin(ang)
                pg.draw.line(surf, col, (int(ox - cx), int(oy - cy)), (int(ex - cx), int(ey - cy)), 1)

        # Draw body segments with simple shading indicating left/right activation
        cx, cy = cam
        segs_iter = list(reversed(self.segs))
        step = 3 if lite_mode else 1
        for idx in range(0, len(segs_iter), step):
            seg = segs_iter[idx]
            # back to front - use worm's own body radius
            r = self.body_radius * (0.9 + 0.12 * math.sin(0.6 * idx))
            x, y = float(seg.pos[0]) - cx, float(seg.pos[1]) - cy
            # base body with worm's color
            body_col = self.color if not getattr(self, 'ready_to_mate', False) else (255, 120, 180)
            pg.draw.circle(surf, body_col, (int(x), int(y)), int(r))
            if not lite_mode:
                # dorsal/ventral shading by left/right activations
                lcol = (255, 120, 110)
                rcol = (210, 220, 230)
                la = int(255 * seg.left.activation)
                ra = int(255 * seg.right.activation)
                # left/right small arcs
                angle = seg.dir
                lx = x + 0.5 * r * math.cos(angle + math.pi / 2)
                ly = y + 0.5 * r * math.sin(angle + math.pi / 2)
                rx = x + 0.5 * r * math.cos(angle - math.pi / 2)
                ry = y + 0.5 * r * math.sin(angle - math.pi / 2)
                pg.draw.circle(surf, (lcol[0], max(0, lcol[1] - (255 - la)), max(0, lcol[2] - (255 - la))), (int(lx), int(ly)), int(max(3, 0.35 * r)))
                pg.draw.circle(surf, (rcol[0], max(0, rcol[1] - (255 - ra)), max(0, rcol[2] - (255 - ra))), (int(rx), int(ry)), int(max(3, 0.35 * r)))

        # Draw DNA tag above the head
        if draw_dna:
            self.draw_dna_tag(surf, cam)


class Slider:
    def __init__(self, rect: pg.Rect, min_v: float, max_v: float, value: float, label: str, fmt: str = "{:.2f}", on_change: Callable[[float], None] | None = None):
        self.rect = rect
        self.min_v = min_v
        self.max_v = max_v
        self.value = float(value)
        self.label = label
        self.fmt = fmt
        self.on_change = on_change
        self.dragging = False

    def set_value(self, v: float):
        self.value = max(self.min_v, min(self.max_v, v))
        if self.on_change:
            self.on_change(self.value)

    def handle_event(self, ev: pg.event.Event) -> None:
        if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.dragging = True
                self._update_from_pos(ev.pos)
        elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pg.MOUSEMOTION and self.dragging:
            self._update_from_pos(ev.pos)

    def _update_from_pos(self, pos: Tuple[int, int]) -> None:
        x, _ = pos
        t = (x - self.rect.x) / max(1, self.rect.w)
        self.set_value(self.min_v + t * (self.max_v - self.min_v))

    def draw(self, surf: pg.Surface, font: pg.font.Font) -> None:
        # Bar - fully rounded pill shape
        radius = self.rect.h // 2
        pg.draw.rect(surf, (30, 40, 70), self.rect, border_radius=radius)
        # Fill - also fully rounded
        t = (self.value - self.min_v) / (self.max_v - self.min_v)
        fill_w = max(self.rect.h, int(self.rect.w * t))  # Ensure minimum width for rounded corners
        fill = pg.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pg.draw.rect(surf, (80, 120, 200), fill, border_radius=radius)
        # Handle - larger circle
        hx = self.rect.x + int(self.rect.w * t)
        hy = self.rect.centery
        pg.draw.circle(surf, (200, 220, 255), (hx, hy), int(self.rect.h * 0.7))
        pg.draw.circle(surf, (160, 180, 220), (hx, hy), int(self.rect.h * 0.7), 1)
        # Label
        text = f"{self.label}: {self.fmt.format(self.value)}"
        img = font.render(text, True, (230, 235, 245))
        surf.blit(img, (self.rect.x, self.rect.y - 18))


def _get_attr_by_path(root_map: dict, path: str):
    head, *rest = path.split('.')
    obj = root_map.get(head)
    for name in rest:
        obj = getattr(obj, name)
    return obj


def _set_attr_by_path(root_map: dict, path: str, value):
    parts = path.split('.')
    head = parts[0]
    obj = root_map.get(head)
    for name in parts[1:-1]:
        obj = getattr(obj, name)
    setattr(obj, parts[-1], value)


def build_sliders(params: Params, worm: Worm, maze: Maze, start: Tuple[int, int]) -> List[Slider]:
    x, y = start
    w = 220
    h = 12
    gap = 42
    sliders: List[Slider] = []

    def add(label: str, min_v: float, max_v: float, val_get: Callable[[], float], val_set: Callable[[float], None], fmt: str = "{:.2f}"):
        nonlocal y
        s = Slider(pg.Rect(x, y, w, h), min_v, max_v, val_get(), label, fmt, on_change=val_set)
        sliders.append(s)
        y += gap

    # Load slider config if present
    cfg_path = 'pygame_worm/config.json'
    loaded_from_config = False
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        roots = { 'params': params, 'worm': worm, 'maze': maze }
        callbacks = {
            'apply_muscle_params': lambda: worm.apply_muscle_params(params),
            'regenerate_maze': lambda: maze.regenerate(),
        }
        for section in cfg.get('sections', []):
            for item in section.get('items', []):
                label = item.get('label', 'Param')
                vmin = float(item.get('min', 0.0))
                vmax = float(item.get('max', 1.0))
                paths = item.get('paths', [])
                ptype = item.get('type', 'float')
                fmt = item.get('fmt', '{:.2f}')
                post = item.get('post')
                def make_get(paths=paths, roots=roots, ptype=ptype):
                    def _get():
                        v = _get_attr_by_path(roots, paths[0])
                        return float(v) if ptype == 'float' else float(int(v))
                    return _get
                def make_set(paths=paths, roots=roots, ptype=ptype, post=post):
                    def _set(v):
                        vv = float(v)
                        if ptype == 'int':
                            vv = int(round(vv))
                        for p in paths:
                            _set_attr_by_path(roots, p, vv)
                        if post and post in callbacks:
                            callbacks[post]()
                    return _set
                sliders.append(Slider(pg.Rect(x, y, w, h), vmin, vmax, make_get()(), label, fmt, on_change=make_set()))
                y += gap
        loaded_from_config = True
    except Exception:
        loaded_from_config = False

    if not loaded_from_config:
        # Fallback to built-in sliders if config is missing/broken
        # Motion
        add("Wave Freq (Hz)", 0.2, 3.0, lambda: params.wave_freq, lambda v: setattr(params, "wave_freq", float(v)))
        add("Wave Amp", 0.0, 1.0, lambda: params.wave_amp, lambda v: setattr(params, "wave_amp", float(v)))
        add("Forward Gain", 5.0, 80.0, lambda: params.forward_gain, lambda v: setattr(params, "forward_gain", float(v)))

        # Sensors
        add("Steer Gain", 0.0, 3.0, lambda: params.steer_gain, lambda v: setattr(params, "steer_gain", float(v)))
        add("Sensor FOV (deg)", 5.0, 90.0, lambda: params.sensor_fov_deg, lambda v: setattr(params, "sensor_fov_deg", float(v)), fmt="{:.0f}")
        add("Sensor Range", 40.0, 260.0, lambda: params.sensor_range, lambda v: setattr(params, "sensor_range", float(v)), fmt="{:.0f}")
        add("Avoid Dist", 10.0, 220.0, lambda: params.avoidance_threshold, lambda v: setattr(params, "avoidance_threshold", float(v)), fmt="{:.0f}")

        # Neurons
        add("Head k", 0.4, 3.0, lambda: params.k_gain_head, lambda v: [setattr(params, "k_gain_head", float(v)), setattr(worm.head_neuron_L, "k_gain", float(v)), setattr(worm.head_neuron_R, "k_gain", float(v))] and None)
        add("Head interval", 80.0, 800.0, lambda: float(params.head_interval_ms), lambda v: [setattr(params, "head_interval_ms", int(v)), setattr(worm.head_neuron_L, "interval_ms", int(v)), setattr(worm.head_neuron_R, "interval_ms", int(v))] and None, fmt="{:.0f}")
        add("BG k", 0.4, 3.0, lambda: params.k_gain_bg, lambda v: [setattr(params, "k_gain_bg", float(v)), setattr(worm.bg_neuron, "k_gain", float(v))] and None)
        add("BG interval", 80.0, 1200.0, lambda: float(params.bg_interval_ms), lambda v: [setattr(params, "bg_interval_ms", int(v)), setattr(worm.bg_neuron, "interval_ms", int(v))] and None, fmt="{:.0f}")

    # Muscles
    def apply_muscles(_: float) -> None:
        worm.apply_muscle_params(params)

    add("Muscle baseline", 0.0, 0.3, lambda: params.muscle_baseline, lambda v: [setattr(params, "muscle_baseline", float(v)), apply_muscles(v)] and None)
    add("Muscle gain", 0.5, 4.0, lambda: params.muscle_gain, lambda v: [setattr(params, "muscle_gain", float(v)), apply_muscles(v)] and None)
    add("Muscle gamma", 0.4, 3.0, lambda: params.muscle_gamma, lambda v: [setattr(params, "muscle_gamma", float(v)), apply_muscles(v)] and None)

    return sliders


class ScrollableSliderPanel:
    def __init__(self, rect: pg.Rect, sliders: List[Slider]):
        self.rect = rect
        self.sliders = sliders
        self.scroll_offset = 0
        self.max_scroll = 0
        self.dragging_scrollbar = False
        self.update_max_scroll()

    def update_max_scroll(self) -> None:
        if not self.sliders:
            self.max_scroll = 0
            return
        # Calculate total content height
        max_y = max((s.rect.y + s.rect.h + 20) for s in self.sliders)
        min_y = min(s.rect.y for s in self.sliders) if self.sliders else 0
        content_height = max_y - min_y
        self.max_scroll = max(0, content_height - self.rect.h)

    def handle_event(self, ev: pg.event.Event) -> None:
        if ev.type == pg.MOUSEWHEEL:
            # Check if mouse is over the panel
            mouse_pos = pg.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                # Scroll up/down
                self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset - ev.y * 20))
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
            # Check scrollbar click
            scrollbar_rect = self._get_scrollbar_rect()
            if scrollbar_rect and scrollbar_rect.collidepoint(ev.pos):
                self.dragging_scrollbar = True
            else:
                # Pass to sliders with adjusted coordinates
                adjusted_ev = self._adjust_event(ev)
                if adjusted_ev:
                    for s in self.sliders:
                        s.handle_event(adjusted_ev)
        elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
            self.dragging_scrollbar = False
            adjusted_ev = self._adjust_event(ev)
            if adjusted_ev:
                for s in self.sliders:
                    s.handle_event(adjusted_ev)
        elif ev.type == pg.MOUSEMOTION:
            if self.dragging_scrollbar:
                # Update scroll based on mouse position
                scrollbar_rect = self._get_scrollbar_track_rect()
                if scrollbar_rect:
                    rel_y = ev.pos[1] - scrollbar_rect.y
                    scroll_ratio = rel_y / max(1, scrollbar_rect.h)
                    self.scroll_offset = max(0, min(self.max_scroll, scroll_ratio * self.max_scroll))
            else:
                adjusted_ev = self._adjust_event(ev)
                if adjusted_ev:
                    for s in self.sliders:
                        s.handle_event(adjusted_ev)

    def _adjust_event(self, ev: pg.event.Event) -> pg.event.Event | None:
        """Adjust event coordinates for scrolling and clip to panel bounds."""
        if not hasattr(ev, 'pos'):
            return ev
        x, y = ev.pos
        # Check if within panel bounds
        if not self.rect.collidepoint((x, y)):
            return None
        # Adjust y coordinate for scroll offset
        adjusted_y = y + self.scroll_offset
        # Create new event with adjusted position
        new_ev = pg.event.Event(ev.type, {'pos': (x, adjusted_y), 'button': getattr(ev, 'button', None)})
        return new_ev

    def _get_scrollbar_track_rect(self) -> pg.Rect | None:
        """Get the scrollbar track rectangle."""
        if self.max_scroll <= 0:
            return None
        return pg.Rect(self.rect.right - 12, self.rect.y, 10, self.rect.h)

    def _get_scrollbar_rect(self) -> pg.Rect | None:
        """Get the scrollbar handle rectangle."""
        if self.max_scroll <= 0:
            return None
        track = self._get_scrollbar_track_rect()
        if not track:
            return None
        # Calculate handle size and position
        visible_ratio = self.rect.h / max(1, self.rect.h + self.max_scroll)
        handle_h = max(20, int(track.h * visible_ratio))
        scroll_ratio = self.scroll_offset / max(1, self.max_scroll)
        handle_y = track.y + int((track.h - handle_h) * scroll_ratio)
        return pg.Rect(track.x, handle_y, track.w, handle_h)

    def draw(self, surf: pg.Surface, font: pg.font.Font) -> None:
        # Create a subsurface for clipping
        clip_rect = self.rect.clip(surf.get_rect())
        if clip_rect.w <= 0 or clip_rect.h <= 0:
            return
        
        # Save original clip
        original_clip = surf.get_clip()
        surf.set_clip(clip_rect)
        
        # Draw sliders with offset
        for s in self.sliders:
            # Adjust slider position for scrolling
            original_y = s.rect.y
            s.rect.y = original_y - self.scroll_offset
            # Only draw if visible in panel
            if s.rect.y + s.rect.h >= self.rect.y and s.rect.y <= self.rect.y + self.rect.h:
                s.draw(surf, font)
            s.rect.y = original_y
        
        # Restore original clip
        surf.set_clip(original_clip)
        
        # Draw scrollbar if needed - fully rounded pill shapes
        if self.max_scroll > 0:
            track = self._get_scrollbar_track_rect()
            handle = self._get_scrollbar_rect()
            if track and handle:
                # Track - fully rounded
                pg.draw.rect(surf, (30, 35, 50), track, border_radius=track.w // 2)
                # Handle - fully rounded pill
                pg.draw.rect(surf, (80, 100, 140), handle, border_radius=handle.w // 2)


def create_random_genetics(base_params: Params | None = None, variation: float = 0.3, rng=None) -> Params:
    """Create a new Params with random genetics across full ranges.

    Args:
        base_params: Base parameters to copy from (None = use defaults)
        variation: Unused; kept for API compatibility.
        rng: Random number generator to use (uses global random if None)
    """
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


class Button:
    def __init__(self, rect: pg.Rect, label: str, get_on: Callable[[], bool], on_toggle: Callable[[bool], None]):
        self.rect = rect
        self.label = label
        self.get_on = get_on
        self.on_toggle = on_toggle

    def handle_event(self, ev: pg.event.Event) -> None:
        if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.on_toggle(not self.get_on())

    def draw(self, surf: pg.Surface, font: pg.font.Font) -> None:
        on = self.get_on()
        bg = (24, 60, 36) if on else (60, 24, 24)
        bd = (42, 120, 70) if on else (120, 60, 60)
        # Fully rounded pill shape button
        radius = self.rect.h // 2
        pg.draw.rect(surf, bg, self.rect, border_radius=radius)
        pg.draw.rect(surf, bd, self.rect, width=2, border_radius=radius)
        text = f"{self.label}: {'On' if on else 'Off'}"
        img = font.render(text, True, (230, 235, 245))
        surf.blit(img, (self.rect.x + 10, self.rect.y + 6))


def run() -> None:
    pg.init()
    pg.display.set_caption("Worm — Neuron & Muscle Fibers (Pygame) - 100 Worms")
    screen = pg.display.set_mode(WIN_SIZE)
    clock = pg.time.Clock()
    # Optional GPU renderer (ModernGL) toggle via --gl or GL=1
    import os as _os
    use_gl = ('--gl' in sys.argv) or (_os.environ.get('GL', '0') == '1')
    if use_gl:
        try:
            pg.display.set_mode(WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
            import moderngl  # type: ignore
            from .gl_renderer import GLRenderer, CircleInstance
            ctx = moderngl.create_context()
            glr = GLRenderer(ctx, WIN_SIZE)
            screen = None
        except Exception as e:
            print("Falling back to CPU renderer:", e, file=sys.stderr)
            use_gl = False

    maze = Maze(WORLD_SIZE, margin=20)
    base_params = Params()  # Base genetics
    
    # Create 100 worms with random positions, headings, colors, and UNIQUE GENETICS
    import random
    import colorsys
    random.seed()  # Use system time for randomness
    worms: List[Worm] = []
    for i in range(100):
        # Random position in world
        x = random.uniform(WORLD_SIZE[0] * 0.1, WORLD_SIZE[0] * 0.9)
        y = random.uniform(WORLD_SIZE[1] * 0.1, WORLD_SIZE[1] * 0.9)
        heading = random.uniform(0, 2 * math.pi)
        # Generate varied colors (pastel palette)
        hue = random.uniform(0, 360)
        # Convert HSV to RGB for varied colors
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.3 + random.uniform(0, 0.4), 0.85 + random.uniform(0, 0.15))
        color = (int(r * 255), int(g * 255), int(b * 255))
        # Each worm gets unique genetic parameters! (50% variation for obvious differences)
        worm_genetics = create_random_genetics(base_params, variation=0.5, rng=random)
        # Physical diversity based on genetics
        num_segments = int(BASE_SEGMENTS * random.uniform(0.6, 1.4))  # 11-25 segments
        body_radius = BASE_BODY_RADIUS * random.uniform(0.6, 1.5)  # 6-15 pixels
        seg_spacing = BASE_SEG_SPACING * random.uniform(0.7, 1.3)  # 11-21 pixels
        worms.append(Worm(origin=(x, y), heading=heading, params=worm_genetics, color=color,
                         segments=num_segments, body_radius=body_radius, seg_spacing=seg_spacing))
    
    # First worm is the "main" one for UI control
    main_worm = worms[0]
    
    # DEBUG: Write genetics to file to verify diversity
    with open('worm_genetics_debug.txt', 'w') as f:
        f.write("=== GENETICS VERIFICATION ===\n")
        for i in range(min(10, len(worms))):
            w = worms[i]
            f.write(f"Worm {i}:\n")
            f.write(f"  wave_freq={w.params.wave_freq:.3f}, forward_gain={w.params.forward_gain:.1f}\n")
            f.write(f"  wave_amp={w.params.wave_amp:.3f}, joint_flex={w.params.joint_flex:.3f}\n")
            f.write(f"  segments={w.num_segments}, radius={w.body_radius:.1f}, spacing={w.seg_spacing:.1f}\n")
            f.write(f"  steer_gain={w.params.steer_gain:.3f}, sensor_range={w.params.sensor_range:.1f}\n\n")
        f.write("=============================\n")
    print("Genetics written to worm_genetics_debug.txt")

    font = pg.font.SysFont(None, 18)
    big_font = pg.font.SysFont(None, 22)
    sliders = build_sliders(main_worm.params, main_worm, maze, start=(WIN_SIZE[0] - 260, 120))
    # Create scrollable panel for sliders
    slider_panel_rect = pg.Rect(WIN_SIZE[0] - 270, 110, 250, WIN_SIZE[1] - 130)
    slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)
    # Buttons
    neuron_btn = Button(pg.Rect(WIN_SIZE[0] - 260, 70, 180, 28), "Neurons (Main Worm)", get_on=lambda: main_worm.params.neurons_enabled, on_toggle=lambda v: setattr(main_worm.params, "neurons_enabled", bool(v)))

    # UI visibility toggle
    show_ui = False
    # Camera mode: False = follow main worm, True = overview
    camera_overview = True
    # Manual camera position (used in overview mode)
    manual_cam_target = np.array([WORLD_SIZE[0] * 0.5, WORLD_SIZE[1] * 0.5], dtype=float)
    camera_speed = 800.0  # pixels per second

    t_sec = 0.0
    # Camera top-left in world coordinates
    cam = np.array([0.0, 0.0], dtype=float)
    # Food spawn scheduler
    import random as _random
    def _schedule_next_food_spawn(now: float) -> float:
        return now + _random.uniform(FOOD_SPAWN_MIN_S, FOOD_SPAWN_MAX_S)
    next_food_spawn_t = _schedule_next_food_spawn(t_sec)
    # Stats logging setup
    import os, csv
    from datetime import datetime
    stats_dir = os.path.join('.', 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    initial_food_total = sum(1 for f in maze.foods if not f.eaten)
    food_spawned_count = 0
    stats_log = {
        't': [],
        'pop': [],
        'ready': [],
        'births': [],
        'deaths': [],
        'food_avail': [],
        'food_total': [],
        'food_ratio': [],
        'avg_spd': [],
        'max_spd': [],
    }
    # Downsampled stats tracking
    globals()['_last_stats_t'] = -1.0
    # Life-sim counters
    births_count = 0
    deaths_count = 0
    running = True
    while running:
        dt_ms = clock.tick(TARGET_FPS)
        dt = dt_ms / 1000.0
        t_sec += dt

        # Minimap rectangle (for hit testing and draw)
        mini_w = 240
        mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
        mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False
                elif ev.key == pg.K_SPACE:
                    # Pause all worms
                    for w in worms:
                        w.paused = not w.paused
                elif ev.key == pg.K_r:
                    maze.regenerate()
                elif ev.key == pg.K_c:
                    # Reload slider config and rebuild sliders for current main worm
                    sliders = build_sliders(main_worm.params, main_worm, maze, start=(WIN_SIZE[0] - 260, 120))
                    slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)
                elif ev.key == pg.K_h:
                    show_ui = not show_ui
                elif ev.key == pg.K_v:
                    # Toggle camera view
                    camera_overview = not camera_overview
                    if camera_overview:
                        # Reset to center when switching to overview
                        manual_cam_target = np.array([WORLD_SIZE[0] * 0.5, WORLD_SIZE[1] * 0.5], dtype=float)
                elif ev.key == pg.K_s:
                    # Stop simulation and show final statistics
                    running = False
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                # Click on minimap: teleport camera to world position or nearest worm
                if mini_rect.collidepoint(ev.pos):
                    sx_inv = WORLD_SIZE[0] / max(1, mini_rect.w)
                    sy_inv = WORLD_SIZE[1] / max(1, mini_rect.h)
                    world_x = (ev.pos[0] - mini_rect.x) * sx_inv
                    world_y = (ev.pos[1] - mini_rect.y) * sy_inv
                    # Snap to nearest worm if close in world space (threshold derived from 6 mini pixels)
                    threshold_world = 6 * sx_inv
                    nearest = None
                    best_d = 1e18
                    for w in worms:
                        wx, wy = w.head_pos()
                        d = math.hypot(wx - world_x, wy - world_y)
                        if d < best_d:
                            best_d = d
                            nearest = (wx, wy)
                    if nearest is not None and best_d <= threshold_world:
                        world_x, world_y = nearest
                    camera_overview = True
                    manual_cam_target = np.array([world_x, world_y], dtype=float)
                elif ( not (show_ui and slider_panel_rect.collidepoint(ev.pos))) and any((getattr(w, 'last_tag_rect', None) is not None) and getattr(w,'last_tag_rect').collidepoint(ev.pos) for w in worms):
                    for w in worms:
                        r = getattr(w, 'last_tag_rect', None)
                        if r is not None and r.collidepoint(ev.pos):
                            main_worm = w
                            sliders = build_sliders(main_worm.params, main_worm, maze, start=(WIN_SIZE[0] - 260, 120))
                            slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)
                            show_ui = True
                            break
                # Click on main viewport in overview mode: move camera target directly
                elif camera_overview and (ev.pos[0] < WIN_SIZE[0] - 300 or not show_ui):
                    world_x = ev.pos[0] + cam[0]
                    world_y = ev.pos[1] + cam[1]
                    manual_cam_target = np.array([world_x, world_y], dtype=float)
            # Buttons & Sliders
            if show_ui:
                neuron_btn.handle_event(ev)
                slider_panel.handle_event(ev)
        
        # Arrow key camera control in overview mode
        if camera_overview:
            keys = pg.key.get_pressed()
            move_speed = camera_speed * dt
            if keys[pg.K_LEFT] or keys[pg.K_a]:
                manual_cam_target[0] -= move_speed
            if keys[pg.K_RIGHT] or keys[pg.K_d]:
                manual_cam_target[0] += move_speed
            if keys[pg.K_UP] or keys[pg.K_w]:
                manual_cam_target[1] -= move_speed
            if keys[pg.K_DOWN] or keys[pg.K_s]:
                manual_cam_target[1] += move_speed
            # Clamp manual camera target to world bounds
            manual_cam_target[0] = max(WIN_SIZE[0] * 0.5, min(WORLD_SIZE[0] - WIN_SIZE[0] * 0.5, manual_cam_target[0]))
            manual_cam_target[1] = max(WIN_SIZE[1] * 0.5, min(WORLD_SIZE[1] - WIN_SIZE[1] * 0.5, manual_cam_target[1]))

        # Update all worms (each uses its own genetic parameters)
        for w in worms:
            w.update(dt_ms, t_sec, maze)

        # Random food spawning (dynamic burst; no auto-despawn)
        if t_sec >= next_food_spawn_t:
            available = maze.available_food()
            target = int(FOOD_TARGET_PER_WORM * len(worms))
            deficit = max(0, target - available)
            nspawn = max(1, min(FOOD_SPAWN_BURST_MAX, deficit))
            spawned = 0
            for _ in range(nspawn):
                if maze.spawn_food_random():
                    spawned += 1
            food_spawned_count += spawned
            next_food_spawn_t = _schedule_next_food_spawn(t_sec)



        # Cull dead worms
        if any(not w.alive for w in worms):
            removed = sum(1 for w in worms if not w.alive)
            deaths_count += removed
            worms = [w for w in worms if w.alive]
            if not main_worm.alive and len(worms) > 0:
                main_worm = worms[0]
                # Rebuild sliders for new main worm
                sliders = build_sliders(main_worm.params, main_worm, maze, start=(WIN_SIZE[0] - 260, 120))
                slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)

        # Reproduction: grid-accelerated neighbor checks; parents die after mating
        import random as _random
        from .spatial import SpatialHash
        new_worms: List[Worm] = []
        # Build a spatial index over heads
        grid = SpatialHash(cell_size=160.0)
        for w in worms:
            hx, hy = w.head_pos()
            grid.insert(w, hx, hy)
        visited: set[tuple[int, int]] = set()
        for wi in list(worms):
            if not wi.alive:
                continue
            hix, hiy = wi.head_pos()
            # Query neighbors within a local radius
            local_r = 2.0 * max(wi.body_radius, BASE_BODY_RADIUS) * REPRODUCTION_DISTANCE_SCALE + 8.0
            for wj in grid.query_radius(hix, hiy, local_r):
                if wj is wi or not getattr(wj, 'alive', True):
                    continue
                key = (min(id(wi), id(wj)), max(id(wi), id(wj)))
                if key in visited:
                    continue
                visited.add(key)
                hjx, hjy = wj.head_pos()
                d = math.hypot(hix - hjx, hiy - hjy)
                thresh = REPRODUCTION_DISTANCE_SCALE * (wi.body_radius + wj.body_radius)
                if d <= thresh and (wi.ready_to_mate or wj.ready_to_mate):
                    donor = wi if getattr(wi, 'speed_ema', 0.0) >= getattr(wj, 'speed_ema', 0.0) else wj
                    import copy
                    child_params = copy.deepcopy(donor.params)
                    offspring_n = _random.randint(0, 5)
                    for _ in range(offspring_n):
                        num_segments = int(BASE_SEGMENTS * _random.uniform(0.6, 1.4))
                        body_radius = BASE_BODY_RADIUS * _random.uniform(0.6, 1.5)
                        seg_spacing = BASE_SEG_SPACING * _random.uniform(0.7, 1.3)
                        cxp = (hix + hjx) * 0.5 + _random.uniform(-12.0, 12.0)
                        cyp = (hiy + hjy) * 0.5 + _random.uniform(-12.0, 12.0)
                        chead = _random.uniform(0, 2 * math.pi)
                        col = donor.color
                        new_worms.append(Worm(origin=(cxp, cyp), heading=chead, params=child_params, color=col,
                                              segments=num_segments, body_radius=body_radius, seg_spacing=seg_spacing))
                    wi.alive = False
                    wj.alive = False
        if new_worms:
            births_count += len(new_worms)
            worms.extend(new_worms)
        
        # Camera control
        if camera_overview:
            # Center camera on manual target position
            cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], manual_cam_target[0] - WIN_SIZE[0] * 0.5))
            cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], manual_cam_target[1] - WIN_SIZE[1] * 0.5))
        else:
            # Camera follows main worm head, clamped to world
            hx, hy = main_worm.head_pos()
            cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], hx - WIN_SIZE[0] * 0.5))
            cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], hy - WIN_SIZE[1] * 0.5))

        if use_gl:
            glr.begin((cam[0], cam[1]))
            # Obstacles as ellipses
            obs_instances = []
            for obs in maze.obstacles:
                cx = obs.rect.centerx
                cy = obs.rect.centery
                rx = max(1.0, obs.rect.w * 0.5)
                ry = max(1.0, obs.rect.h * 0.5)
                obs_instances.append(CircleInstance(cx, cy, rx, ry, 34/255.0, 45/255.0, 78/255.0))
            glr.draw_instances(obs_instances)
            # Foods near viewport
            vw, vh = WIN_SIZE
            foods = list(maze.foods_near(cam[0] + vw * 0.5, cam[1] + vh * 0.5, max(vw, vh) * 0.8))
            food_r = float(max(1, int(maze.food_radius)))
            food_instances = [
                CircleInstance(f.x, f.y, food_r, food_r, 90/255.0, 200/255.0, 110/255.0)
                for f in foods if not f.eaten
            ]
            glr.draw_instances(food_instances)
            # Worm segments
            lite_mode = len(worms) >= 300
            worm_instances = []
            for i, w in enumerate(worms):
                segs = list(reversed(w.segs))
                step = 3 if lite_mode else 1
                for s_idx in range(0, len(segs), step):
                    seg = segs[s_idx]
                    r = w.body_radius * (0.9 + 0.12 * math.sin(0.6 * s_idx))
                    col = w.color if not getattr(w, 'ready_to_mate', False) else (255, 120, 180)
                    worm_instances.append(CircleInstance(float(seg.pos[0]), float(seg.pos[1]), r, r, col[0]/255.0, col[1]/255.0, col[2]/255.0))
            glr.draw_instances(worm_instances)
            glr.end()
        else:
            maze.draw(screen, (cam[0], cam[1]))
            # Draw all worms (each with their own genetics affecting appearance)
            for i, w in enumerate(worms):
                # Only draw sensors for main worm in follow mode
                draw_sensors = (i == 0 and not camera_overview)
                draw_dna = (i == 0 and not camera_overview) or (len(worms) <= 100)
                lite_mode = len(worms) >= 300
                w.draw(screen, (cam[0], cam[1]), draw_sensors=draw_sensors, draw_dna=draw_dna, lite_mode=lite_mode)

        # Top status bar (avoids minimap area)
        mini_w = 240
        mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
        mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)
        bar_x = 12
        bar_y = 8
        bar_w = max(100, (mini_rect.x - 12) - bar_x)
        bar_h = 48 if show_ui else 66
        bar_rect = pg.Rect(bar_x, bar_y, bar_w, bar_h)
        if not use_gl:
            # Background
            pg.draw.rect(screen, (16, 22, 36), bar_rect, border_radius=10)
            pg.draw.rect(screen, (40, 56, 92), bar_rect, width=1, border_radius=10)
        # Compose metrics
        total_food = len(maze.foods)
        available_food = maze.available_food()
        eaten_food = total_food - available_food
        ratio = (available_food / total_food) if total_food > 0 else 0.0
        ready = sum(1 for w in worms if getattr(w, 'ready_to_mate', False))
        avg_spd = (sum(getattr(w, 'speed_ema', 0.0) for w in worms) / len(worms)) if worms else 0.0
        max_spd = max((getattr(w, 'speed_ema', 0.0) for w in worms), default=0.0)
        # Append time-series stats (sample at ~2 Hz)
        if 'last_stats_t' not in locals():
            last_stats_t = -1.0
        try:
            last_stats_t
        except NameError:
            last_stats_t = -1.0
        if t_sec - (globals().get('_last_stats_t', -1.0)) >= 0.5:
            stats_log['t'].append(float(t_sec))
            stats_log['pop'].append(int(len(worms)))
            stats_log['ready'].append(int(ready))
            stats_log['births'].append(int(births_count))
            stats_log['deaths'].append(int(deaths_count))
            stats_log['food_avail'].append(int(available_food))
            stats_log['food_total'].append(int(total_food))
            stats_log['food_ratio'].append(float(ratio))
            stats_log['avg_spd'].append(float(avg_spd))
            stats_log['max_spd'].append(float(max_spd))
            globals()['_last_stats_t'] = t_sec
        # Next food spawn countdown
        spawn_in = max(0, int(next_food_spawn_t - t_sec))
        if not use_gl:
            # Row 1
            x = bar_x + 10
            y = bar_y + 6
            for txt in [
                f"Pop {len(worms)}",
                f"Ready {ready}",
                f"Births {births_count}",
                f"Deaths {deaths_count}",
                f"Food {available_food}/{total_food} ({ratio:.0%})",
            ]:
                img = font.render(txt, True, (220, 230, 245))
                screen.blit(img, (x, y))
                x += img.get_width() + 18
        # Row 2
        if not use_gl:
            x = bar_x + 10
            y = bar_y + 6 + 18
            mm = int(t_sec) // 60
            ss = int(t_sec) % 60
            for txt in [
                f"Time {mm:02d}:{ss:02d}",
                f"AvgSpd {avg_spd:.0f}",
                f"MaxSpd {max_spd:.0f}",
                f"NextFood {spawn_in}s",
            ]:
                img = font.render(txt, True, (200, 215, 235))
                screen.blit(img, (x, y))
                x += img.get_width() + 18
        # Row 3: hint when UI hidden
        if not use_gl and not show_ui:
            hint_txt = "H: show UI   C: reload   V: view"
            img = font.render(hint_txt, True, (185, 195, 215))
            screen.blit(img, (bar_x + 10, bar_y + 6 + 36))
        
        # Minimap (top-right) only in CPU renderer
        if not use_gl:
            mini_w = 240
            mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
            mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)
            cam_rect = pg.Rect(int(cam[0]), int(cam[1]), WIN_SIZE[0], WIN_SIZE[1])
            maze.draw_minimap(screen, mini_rect, [w.head_pos() for w in worms], cam_rect)
        # HUD panel and hints
        if not use_gl and show_ui:
            panel = pg.Rect(WIN_SIZE[0] - 280, 20, 260, WIN_SIZE[1] - 40)
            # More rounded panel corners
            pg.draw.rect(screen, (16, 20, 36), panel, border_radius=20)
            pg.draw.rect(screen, (40, 50, 80), panel, width=2, border_radius=20)
            title = big_font.render("Controls", True, (210, 220, 240))
            screen.blit(title, (panel.x + 12, panel.y + 10))
            # Small help lines
            help1 = font.render("V: toggle view  H: hide UI  R: new maze", True, (180, 190, 210))
            screen.blit(help1, (panel.x + 12, panel.y + 36))
            help2 = font.render("Arrows/WASD: move camera (overview)", True, (180, 190, 210))
            screen.blit(help2, (panel.x + 12, panel.y + 52))
            help3 = font.render("Click map: jump to location", True, (180, 190, 210))
            screen.blit(help3, (panel.x + 12, panel.y + 68))
            # Worm count
            worm_count = font.render(f"Worms: {len(worms)}", True, (180, 190, 210))
            screen.blit(worm_count, (panel.x + 12, panel.y + 84))
            neuron_btn.draw(screen, font)
            slider_panel.draw(screen, font)
        else:
            # Hint moved into top status bar when UI hidden
            pass

        pg.display.flip()

    # Final statistics overlay (CPU renderer only)
    if not use_gl:
        try:
            screen.fill((10, 14, 26))
            panel = pg.Rect(WIN_SIZE[0]//2 - 360, WIN_SIZE[1]//2 - 160, 720, 320)
            pg.draw.rect(screen, (18, 24, 40), panel, border_radius=18)
            pg.draw.rect(screen, (60, 80, 130), panel, width=2, border_radius=18)
            title = big_font.render("Simulation Summary", True, (230, 240, 255))
            screen.blit(title, (panel.x + 20, panel.y + 16))
            # Compute metrics
            total_time = t_sec
            mm = int(total_time) // 60
            ss = int(total_time) % 60
            final_pop = len(worms)
            available_end = sum(1 for f in maze.foods if not f.eaten)
            total_spawned = food_spawned_count
            initial_total = initial_food_total
            eaten_total = max(0, initial_total + total_spawned - available_end)
            lines = [
                f"Time: {mm:02d}:{ss:02d}",
                f"Final population: {final_pop}",
                f"Births: {births_count}    Deaths: {deaths_count}",
                f"Food: spawned {total_spawned}, eaten {eaten_total}, remaining {available_end}",
            ]
            y = panel.y + 60
            for txt in lines:
                img = font.render(txt, True, (210, 220, 235))
                screen.blit(img, (panel.x + 20, y))
                y += 26
            hint = font.render("Press any key or wait 5s to exit", True, (185, 195, 210))
            screen.blit(hint, (panel.x + 20, panel.y + panel.h - 36))
            pg.display.flip()
            # Wait for short period or input
            wait_ms = 5000
            end_wait = pg.time.get_ticks() + wait_ms
            waiting = True
            while waiting:
                for ev in pg.event.get():
                    if ev.type in (pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN):
                        waiting = False
                        break
                if pg.time.get_ticks() >= end_wait:
                    waiting = False
                pg.time.delay(20)
        except Exception:
            pass

    pg.quit()

    # Persist time-series stats to /stats
    try:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(stats_dir, f"stats_{run_id}.csv")
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            headers = list(stats_log.keys())
            writer.writerow(headers)
            rows = zip(*(stats_log[k] for k in headers))
            for row in rows:
                writer.writerow(row)
        try:
            import matplotlib.pyplot as plt
            series = [
                ("pop", "Population"),
                ("ready", "Ready to mate"),
                ("births", "Births (cumulative)"),
                ("deaths", "Deaths (cumulative)"),
                ("food_avail", "Food available"),
                ("food_total", "Food total"),
                ("food_ratio", "Food availability ratio"),
                ("avg_spd", "Average speed"),
                ("max_spd", "Max speed"),
            ]
            t = stats_log["t"]
            for key, title in series:
                plt.figure(figsize=(8, 3))
                plt.plot(t, stats_log[key], lw=1.6)
                plt.title(title)
                plt.xlabel("Time (s)")
                plt.ylabel(title)
                plt.grid(True, alpha=0.3)
                out_png = os.path.join(stats_dir, f"{key}_{run_id}.png")
                plt.tight_layout()
                plt.savefig(out_png)
                plt.close()
        except Exception:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        pg.quit()
        raise
