from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
import pygame as pg

from .params import Params
from .config import (
    DNA_SPECS,
    DNA_COLORS,
    STARVATION_TIME_S,
    WAVE_PHASE_PER_SEG,
    REPRODUCTION_FOOD_REQUIRED,
)
from .neuro import Neuron, Twitch, compute_twitch_timing
from .brain import Brain
from .muscle import MuscleFiber
from .maze import Maze, Food


@dataclass
class Segment:
    pos: np.ndarray  # shape (2,)
    dir: float       # heading angle radians
    left: MuscleFiber
    right: MuscleFiber


class Worm:
    def __init__(self, origin: Tuple[float, float], heading: float, params: Params,
                 color: Tuple[int, int, int] | None = None,
                 segments: int | None = None,
                 body_radius: float | None = None,
                 seg_spacing: float | None = None):
        self.params = params
        # segments can be overridden by constructor arg; otherwise use genetic value
        self.num_segments = int(segments if segments is not None else getattr(params, 'segments', 6))
        self.body_radius = body_radius if body_radius is not None else 10.0
        self.seg_spacing = seg_spacing if seg_spacing is not None else 16.0

        self.segs: List[Segment] = []
        x, y = origin
        for i in range(self.num_segments):
            pos = np.array([
                x - i * self.seg_spacing * math.cos(heading),
                y - i * self.seg_spacing * math.sin(heading)
            ], dtype=float)
            self.segs.append(Segment(pos=pos, dir=heading, left=MuscleFiber(), right=MuscleFiber()))

        self.head_neuron_L = Neuron(interval_ms=params.head_interval_ms, k_gain=params.k_gain_head)
        self.head_neuron_R = Neuron(interval_ms=params.head_interval_ms, k_gain=params.k_gain_head)
        self.bg_neuron = Neuron(interval_ms=params.bg_interval_ms, k_gain=params.k_gain_bg)
        self.brain = Brain(interval_ms=params.brain_interval_ms)
        self.paused = False
        self.color = color if color is not None else (230, 232, 245)
        self.phase_offset = np.random.uniform(0, 2 * math.pi)
        self.last_sensor_L: float = params.sensor_range
        self.last_sensor_R: float = params.sensor_range
        self.last_sensor_origin_L: Tuple[float, float] = origin
        self.last_sensor_origin_R: Tuple[float, float] = origin
        self.last_sensor_fan: List[Tuple[Tuple[float, float], float, float]] = []
        self.last_food_target: Tuple[float, float] | None = None
        self.current_target: Food | None = None
        self.target_start_t: float = 0.0
        self.target_start_dist: float = float('inf')
        self.target_best_dist: float = float('inf')
        self.avoid_food_until: dict[int, float] = {}
        self.apply_muscle_params(self.params)
        self.alive: bool = True
        self.food_eaten: int = 0
        self.time_since_last_eat: float = 0.0
        self.ready_to_mate: bool = False
        self.last_repro_time: float = -1e6
        self.current_speed: float = 0.0
        self.speed_ema: float = 0.0
        self.max_speed: float = 0.0

    def dna_values(self) -> List[float]:
        vals: List[float] = []
        p = self.params
        for name, mn, mx in DNA_SPECS:
            v = getattr(p, name)
            vf = float(v)
            if mx <= mn:
                n = 0.0
            else:
                n = (vf - mn) / (mx - mn)
            n = 0.0 if n < 0.0 else (1.0 if n > 1.0 else n)
            vals.append(n)
        return vals

    def draw_dna_tag(self, surf: pg.Surface, cam: Tuple[float, float], font: pg.font.Font | None = None) -> None:
        hx, hy = self.head_pos()
        cx, cy = cam
        x = int(hx - cx)
        y = int(hy - cy)
        bars = self.dna_values()
        if not bars:
            return
        bar_w = 4
        gap = 2
        padding = 6
        max_h = 28
        total_w = len(bars) * bar_w + (len(bars) - 1) * gap + 2 * padding
        total_h = max_h + 2 * padding
        tag_x = x - total_w // 2
        tag_y = y - int(self.body_radius) - 10
        bg_rect = pg.Rect(tag_x, tag_y, total_w, total_h)
        pg.draw.rect(surf, (20, 26, 40), bg_rect, border_radius=6)
        pg.draw.rect(surf, (60, 72, 110), bg_rect, width=1, border_radius=6)
        bx = tag_x + 6
        by = tag_y + 6
        for i, n in enumerate(bars):
            h = int(max_h * n)
            col = DNA_COLORS[i % len(DNA_COLORS)]
            pg.draw.rect(surf, col, (bx, by + (max_h - h), 4, h), border_radius=2)
            bx += (4 + 2)
        # Optional text overlay: speed (EMA) and foods eaten
        if font is not None:
            info = f"S:{int(self.speed_ema):d}  E:{int(self.food_eaten):d}"
            img = font.render(info, True, (230, 235, 245))
            surf.blit(img, (tag_x, tag_y - 16))

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
        params = self.params
        if not self.alive or self.paused:
            return
        dt = dt_ms / 1000.0
        self.time_since_last_eat += dt
        if self.time_since_last_eat >= STARVATION_TIME_S:
            self.alive = False
            return

        # Sensor rays from head (throttled)
        hx, hy = self.head_pos()
        hdir = self.head_dir()
        sensor_fov = math.radians(params.sensor_fov_deg)
        if not hasattr(self, "_sensor_frame_mod"):
            self._sensor_frame_mod = np.random.randint(0, 3)
        recompute = (int(t_sec * 60) + self._sensor_frame_mod) % 3 == 0
        if recompute:
            n = max(3, int(params.sensor_fan_rays))
            n = int(max(3, min(n, 9)))
            if n % 2 == 0:
                n += 1
            fan: List[Tuple[Tuple[float, float], float, float]] = []
            for i in range(n):
                t = -1.0 + 2.0 * (i / max(1, n - 1))
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
            self.last_sensor_L = dL
            self.last_sensor_R = dR
            self.last_sensor_origin_L = tipL
            self.last_sensor_origin_R = tipR
        else:
            dL, dR, dC = getattr(self, "_last_dists", (params.sensor_range, params.sensor_range, params.sensor_range))
            self.last_sensor_L = dL
            self.last_sensor_R = dR

        # Compute avoidance bias
        bias = 0.0
        win = max(1.0, params.sensor_end_contact_px)
        rep_x = rep_y = 0.0
        cEndC = max(0.0, min(1.0, (win - dC) / win))
        for (_, ang, dist) in self.last_sensor_fan:
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
        if min(dL, dR) < params.avoidance_threshold:
            bias += 0.5 * params.steer_gain * (dR - dL) / max(1.0, params.sensor_range)
        if not params.neurons_enabled:
            bias = 0.0

        # Food attraction
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
                ang = target_ang - hdir
                while ang > math.pi:
                    ang -= 2 * math.pi
                while ang < -math.pi:
                    ang += 2 * math.pi
                bias += params.attract_gain * (ang / math.pi) * min(1.0, mag)

        # Brain steering + twitches
        if params.neurons_enabled and params.brain_gain > 0.0:
            hx, hy = self.head_pos()
            now = t_sec

            def _is_avoided(f: Food) -> bool:
                until = self.avoid_food_until.get(id(f))
                return until is not None and now < until

            if self.current_target is None or self.current_target.eaten or _is_avoided(self.current_target):
                self.current_target = None
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

            nearest_ang = 0.0
            nearest_dist = float('inf')
            nearest_xy = None
            if self.current_target is not None:
                dx = self.current_target.x - hx
                dy = self.current_target.y - hy
                nearest_dist = math.hypot(dx, dy)
                nearest_ang = math.atan2(dy, dx) - hdir
                nearest_xy = (self.current_target.x, self.current_target.y)
                if nearest_dist < self.target_best_dist:
                    self.target_best_dist = nearest_dist
                elapsed = now - self.target_start_t
                improved = self.target_start_dist - self.target_best_dist
                if (elapsed >= params.brain_stuck_timeout_s and
                    improved < params.brain_stuck_min_improve_px and
                    nearest_dist > self.body_radius * 1.3):
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
            if out.twitch_left > 0.0:
                amp = max(0.0, min(1.0, out.twitch_left * params.brain_twitch_gain))
                r, h, d = compute_twitch_timing(amp)
                self.segs[0].left.trigger(Twitch(target=amp, rise=r, hold=h, decay=d))
            if out.twitch_right > 0.0:
                amp = max(0.0, min(1.0, out.twitch_right * params.brain_twitch_gain))
                r, h, d = compute_twitch_timing(amp)
                self.segs[0].right.trigger(Twitch(target=amp, rise=r, hold=h, decay=d))

        # Drive undulatory gait + steering
        omega = 2.0 * math.pi * params.wave_freq
        for i, seg in enumerate(self.segs):
            phase = omega * t_sec + i * WAVE_PHASE_PER_SEG + self.phase_offset
            avoid_strength = cEndC
            for (_, _, dist) in self.last_sensor_fan:
                avoid_strength = max(avoid_strength, max(0.0, min(1.0, (max(1.0, params.sensor_end_contact_px) - dist) / max(1.0, params.sensor_end_contact_px))))
            amp_scale = 1.0 - 0.5 * avoid_strength
            base = ((math.sin(phase) * params.wave_amp) * amp_scale) if params.neurons_enabled else 0.0
            if params.neurons_enabled:
                seg.left.update()
                seg.right.update()
            left_a = np.clip(base + bias + seg.left.activation, 0.0, 1.0)
            right_a = np.clip(-base - bias + seg.right.activation, 0.0, 1.0)
            turn = (left_a - right_a) * 0.25 * max(0.1, params.joint_flex)
            seg.dir += turn * dt

        # Kinematics and speed meter
        head_speed = 0.0 if not params.neurons_enabled else (
            params.forward_gain * (0.35 + 0.65 * abs(math.sin(omega * t_sec + self.phase_offset)))
        )
        if params.neurons_enabled:
            avoid_strength = cEndC
            for (_, _, dist) in self.last_sensor_fan:
                avoid_strength = max(avoid_strength, max(0.0, min(1.0, (max(1.0, params.sensor_end_contact_px) - dist) / max(1.0, params.sensor_end_contact_px))))
            slow = max(0.0, 1.0 - params.avoid_slow_gain * avoid_strength)
            head_speed *= max(params.min_forward_frac, slow)
        self.current_speed = float(head_speed)
        self.speed_ema = 0.85 * self.speed_ema + 0.15 * self.current_speed
        if self.current_speed > self.max_speed:
            self.max_speed = self.current_speed
        vx = head_speed * math.cos(self.segs[0].dir)
        vy = head_speed * math.sin(self.segs[0].dir)
        self.segs[0].pos += np.array([vx * dt, vy * dt])

        # Border/obstacle response
        if maze.collide_circle(self.head_pos(), self.body_radius):
            turn_sign = 1.0 if dL > dR else -1.0
            self.segs[0].dir += turn_sign * math.radians(60.0)
            ndir = self.segs[0].dir
            step_out = self.body_radius * 0.9
            self.segs[0].pos += np.array([math.cos(ndir) * step_out, math.sin(ndir) * step_out])

        # Follow-the-leader
        for i in range(1, self.num_segments):
            prev = self.segs[i - 1]
            cur = self.segs[i]
            delta = prev.pos - cur.pos
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                continue
            dir_vec = delta / dist
            target = prev.pos - dir_vec * self.seg_spacing
            follow_gain = 10.0 / max(0.2, params.joint_flex)
            cur.pos += (target - cur.pos) * min(1.0, follow_gain * dt)
            cur.dir = math.atan2(prev.pos[1] - cur.pos[1], prev.pos[0] - cur.pos[0])

        # Eat food
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

    def draw(self, surf: pg.Surface, cam: Tuple[float, float], draw_sensors: bool = False,
             draw_dna: bool = False, lite_mode: bool = False, font: pg.font.Font | None = None) -> None:
        params = self.params
        if not self.alive:
            return
        if draw_sensors:
            hx, hy = self.head_pos()
            hdir = self.head_dir()
            sensor_fov = math.radians(params.sensor_fov_deg)
            cx, cy = cam
            col = (160, 190, 255)
            for (origin, ang, dist) in getattr(self, 'last_sensor_fan', []):
                ox, oy = origin
                ex = ox + dist * math.cos(ang)
                ey = oy + dist * math.sin(ang)
                pg.draw.line(surf, col, (int(ox - cx), int(oy - cy)), (int(ex - cx), int(ey - cy)), 1)

        cx, cy = cam
        segs_iter = list(reversed(self.segs))
        step = 3 if lite_mode else 1
        for idx in range(0, len(segs_iter), step):
            seg = segs_iter[idx]
            r = self.body_radius * (0.9 + 0.12 * math.sin(0.6 * idx))
            x, y = float(seg.pos[0]) - cx, float(seg.pos[1]) - cy
            body_col = self.color if not getattr(self, 'ready_to_mate', False) else (255, 120, 180)
            pg.draw.circle(surf, body_col, (int(x), int(y)), int(r))
            if not lite_mode:
                lcol = (255, 120, 110)
                rcol = (210, 220, 230)
                la = int(255 * seg.left.activation)
                ra = int(255 * seg.right.activation)
                angle = seg.dir
                lx = x + 0.5 * r * math.cos(angle + math.pi / 2)
                ly = y + 0.5 * r * math.sin(angle + math.pi / 2)
                rx = x + 0.5 * r * math.cos(angle - math.pi / 2)
                ry = y + 0.5 * r * math.sin(angle - math.pi / 2)
                pg.draw.circle(surf, (lcol[0], max(0, lcol[1] - (255 - la)), max(0, lcol[2] - (255 - la))), (int(lx), int(ly)), int(max(3, 0.35 * r)))
                pg.draw.circle(surf, (rcol[0], max(0, rcol[1] - (255 - ra)), max(0, rcol[2] - (255 - ra))), (int(rx), int(ry)), int(max(3, 0.35 * r)))

        if draw_dna:
            self.draw_dna_tag(surf, cam, font=font)
