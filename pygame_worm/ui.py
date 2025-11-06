from __future__ import annotations

import json
from typing import Callable, List, Tuple
import pygame as pg

from .params import Params
from .worm import Worm
from .maze import Maze


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
        radius = self.rect.h // 2
        pg.draw.rect(surf, (30, 40, 70), self.rect, border_radius=radius)
        t = (self.value - self.min_v) / (self.max_v - self.min_v)
        fill_w = max(self.rect.h, int(self.rect.w * t))
        fill = pg.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pg.draw.rect(surf, (80, 120, 200), fill, border_radius=radius)
        hx = self.rect.x + int(self.rect.w * t)
        hy = self.rect.centery
        pg.draw.circle(surf, (200, 220, 255), (hx, hy), int(self.rect.h * 0.7))
        pg.draw.circle(surf, (160, 180, 220), (hx, hy), int(self.rect.h * 0.7), 1)
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
        add("Wave Freq (Hz)", 0.2, 3.0, lambda: params.wave_freq, lambda v: setattr(params, "wave_freq", float(v)))
        add("Wave Amp", 0.0, 1.0, lambda: params.wave_amp, lambda v: setattr(params, "wave_amp", float(v)))
        add("Forward Gain", 5.0, 80.0, lambda: params.forward_gain, lambda v: setattr(params, "forward_gain", float(v)))
        add("Steer Gain", 0.0, 3.0, lambda: params.steer_gain, lambda v: setattr(params, "steer_gain", float(v)))
        add("Sensor FOV (deg)", 5.0, 90.0, lambda: params.sensor_fov_deg, lambda v: setattr(params, "sensor_fov_deg", float(v)), fmt="{:.0f}")
        add("Sensor Range", 40.0, 260.0, lambda: params.sensor_range, lambda v: setattr(params, "sensor_range", float(v)), fmt="{:.0f}")
        add("Avoid Dist", 10.0, 220.0, lambda: params.avoidance_threshold, lambda v: setattr(params, "avoidance_threshold", float(v)), fmt="{:.0f}")
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
        max_y = max((s.rect.y + s.rect.h + 20) for s in self.sliders)
        min_y = min(s.rect.y for s in self.sliders) if self.sliders else 0
        content_height = max_y - min_y
        self.max_scroll = max(0, content_height - self.rect.h)

    def _adjust_event(self, ev: pg.event.Event) -> pg.event.Event | None:
        if not hasattr(ev, 'pos'):
            return ev
        x, y = ev.pos
        if not self.rect.collidepoint((x, y)):
            return None
        adj_pos = (x, y + self.scroll_offset)
        new_ev = pg.event.Event(ev.type, {k: (adj_pos if k == 'pos' else getattr(ev, k)) for k in ev.__dict__ if not k.startswith('_')})
        return new_ev

    def _get_scrollbar_track_rect(self) -> pg.Rect | None:
        if self.max_scroll <= 0:
            return None
        return pg.Rect(self.rect.right - 10, self.rect.y, 8, self.rect.h)

    def _get_scrollbar_rect(self) -> pg.Rect | None:
        track = self._get_scrollbar_track_rect()
        if not track or self.max_scroll <= 0:
            return None
        handle_h = max(24, int(self.rect.h * (self.rect.h / (self.rect.h + self.max_scroll))))
        t = 0 if self.max_scroll == 0 else (self.scroll_offset / self.max_scroll)
        handle_y = int(track.y + t * (track.h - handle_h))
        return pg.Rect(track.x, handle_y, track.w, handle_h)

    def handle_event(self, ev: pg.event.Event) -> None:
        if ev.type == pg.MOUSEWHEEL:
            mouse_pos = pg.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset - ev.y * 20))
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
            scrollbar_rect = self._get_scrollbar_rect()
            if scrollbar_rect and scrollbar_rect.collidepoint(ev.pos):
                self.dragging_scrollbar = True
            else:
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
        elif ev.type == pg.MOUSEMOTION and self.dragging_scrollbar:
            track = self._get_scrollbar_track_rect()
            if track:
                rel = (ev.pos[1] - track.y) / max(1, track.h)
                self.scroll_offset = max(0, min(self.max_scroll, int(rel * (self.max_scroll))))

    def draw(self, surf: pg.Surface, font: pg.font.Font) -> None:
        offset_surf = surf.subsurface(self.rect)
        clip_rect = pg.Rect(0, 0, self.rect.w, self.rect.h)
        for s in self.sliders:
            r = s.rect.move(0, -self.scroll_offset)
            s_vis = Slider(r, s.min_v, s.max_v, s.value, s.label, s.fmt, s.on_change)
            s_vis.draw(offset_surf, font)
        if self.max_scroll > 0:
            track = self._get_scrollbar_track_rect()
            handle = self._get_scrollbar_rect()
            if track and handle:
                pg.draw.rect(surf, (30, 35, 50), track, border_radius=track.w // 2)
                pg.draw.rect(surf, (80, 100, 140), handle, border_radius=handle.w // 2)


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
        radius = self.rect.h // 2
        pg.draw.rect(surf, bg, self.rect, border_radius=radius)
        pg.draw.rect(surf, bd, self.rect, width=2, border_radius=radius)
        text = f"{self.label}: {'On' if on else 'Off'}"
        img = font.render(text, True, (230, 235, 245))
        surf.blit(img, (self.rect.x + 10, self.rect.y + 6))

