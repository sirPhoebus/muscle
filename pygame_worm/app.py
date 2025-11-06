from __future__ import annotations

import sys
import math
import random
from typing import List, Tuple

import pygame as pg
import numpy as np

from .config import (
    WIN_SIZE, WORLD_SIZE, TARGET_FPS,
    BASE_SEGMENTS, BASE_BODY_RADIUS, BASE_SEG_SPACING,
    FOOD_SPAWN_MIN_S, FOOD_SPAWN_MAX_S, FOOD_TARGET_PER_WORM, FOOD_SPAWN_BURST_MAX,
    REPRODUCTION_DISTANCE_SCALE,
    DNA_COLORS,
)
from .maze import Maze
from .params import Params, create_random_genetics
from .worm import Worm
from .ui import build_sliders, ScrollableSliderPanel, Button
from .gl_renderer import GLRenderer, CircleInstance
from .spatial import SpatialHash


def _init_window() -> Tuple[pg.Surface | None, GLRenderer | None, bool]:
    pg.display.set_caption("Worm — Neuron & Muscle Fibers")
    screen: pg.Surface | None = pg.display.set_mode(WIN_SIZE)
    use_gl = ('--gl' in sys.argv) or (pg.os.environ.get('GL', '0') == '1')
    if not use_gl:
        return screen, None, False
    try:
        pg.display.set_mode(WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        import moderngl  # type: ignore
        ctx = moderngl.create_context()
        return None, GLRenderer(ctx, WIN_SIZE), True
    except Exception as e:
        print("Falling back to CPU renderer:", e, file=sys.stderr)
        screen = pg.display.set_mode(WIN_SIZE)
        return screen, None, False


def _draw_cpu(screen: pg.Surface, maze: Maze, worms: List[Worm], cam: Tuple[float, float], camera_overview: bool,
              font: pg.font.Font, big_font: pg.font.Font, show_ui: bool, slider_panel: ScrollableSliderPanel, neuron_btn: Button,
              t_sec: float, next_food_spawn_t: float, births_count: int, deaths_count: int,
              sim_speed: float,
              selected_worm: Worm | None = None) -> None:
    maze.draw(screen, cam)
    for i, w in enumerate(worms):
        draw_sensors = (i == 0 and not camera_overview)
        draw_dna = (i == 0 and not camera_overview) or (len(worms) <= 100)
        lite_mode = len(worms) >= 300
        w.draw(screen, cam, draw_sensors=draw_sensors, draw_dna=draw_dna, lite_mode=lite_mode, font=font)

    # HUD (CPU mode only)
    mini_w = 240
    mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
    mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)
    bar_x = 12
    bar_y = 8
    bar_w = max(100, (mini_rect.x - 12) - bar_x)
    bar_h = 48 if show_ui else 66
    bar_rect = pg.Rect(bar_x, bar_y, bar_w, bar_h)
    pg.draw.rect(screen, (16, 22, 36), bar_rect, border_radius=10)
    pg.draw.rect(screen, (40, 56, 92), bar_rect, width=1, border_radius=10)
    total_food = len(maze.foods)
    available_food = maze.available_food()
    ratio = (available_food / total_food) if total_food > 0 else 0.0
    ready = sum(1 for w in worms if getattr(w, 'ready_to_mate', False))
    avg_spd = (sum(getattr(w, 'speed_ema', 0.0) for w in worms) / len(worms)) if worms else 0.0
    max_spd = max((getattr(w, 'speed_ema', 0.0) for w in worms), default=0.0)
    spawn_in = max(0, int(next_food_spawn_t - t_sec))
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
    x = bar_x + 10
    y = bar_y + 6 + 18
    mm = int(t_sec) // 60
    ss = int(t_sec) % 60
    for txt in [
        f"Time {mm:02d}:{ss:02d}",
        f"AvgSpd {avg_spd:.0f}",
        f"MaxSpd {max_spd:.0f}",
        f"NextFood {spawn_in}s",
        f"Speed x{sim_speed:.2f}",
    ]:
        img = font.render(txt, True, (200, 215, 235))
        screen.blit(img, (x, y))
        x += img.get_width() + 18
    if not show_ui:
        hint_txt = "H: show UI   C: reload   V: view"
        img = font.render(hint_txt, True, (185, 195, 215))
        screen.blit(img, (bar_x + 10, bar_y + 6 + 36))
    mini_w = 240
    mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
    mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)
    cam_rect = pg.Rect(int(cam[0]), int(cam[1]), WIN_SIZE[0], WIN_SIZE[1])
    maze.draw_minimap(screen, mini_rect, [w.head_pos() for w in worms], cam_rect, [getattr(w, 'ready_to_mate', False) for w in worms])
    if show_ui:
        panel = pg.Rect(WIN_SIZE[0] - 280, 20, 260, WIN_SIZE[1] - 40)
        pg.draw.rect(screen, (16, 20, 36), panel, border_radius=20)
        pg.draw.rect(screen, (40, 50, 80), panel, width=2, border_radius=20)
        title = big_font.render("Controls", True, (210, 220, 240))
        screen.blit(title, (panel.x + 12, panel.y + 10))
        help1 = font.render("V: toggle view  H: hide UI  R: new maze", True, (180, 190, 210))
        screen.blit(help1, (panel.x + 12, panel.y + 36))
        help2 = font.render("Arrows/WASD: move camera (overview)", True, (180, 190, 210))
        screen.blit(help2, (panel.x + 12, panel.y + 52))
        help3 = font.render("Click worm: select   [ / ]: cycle   F: follow", True, (180, 190, 210))
        screen.blit(help3, (panel.x + 12, panel.y + 68))
        worm_count = font.render(f"Worms: {len(worms)}", True, (180, 190, 210))
        screen.blit(worm_count, (panel.x + 12, panel.y + 84))
        neuron_btn.draw(screen, font)
        slider_panel.draw(screen, font)

    # Selected worm info card at top-right (below minimap)
    if selected_worm is not None:
        card_w = 260
        card_x = WIN_SIZE[0] - card_w - 12
        card_y = mini_rect.bottom + 10
        card_h = 120
        panel = pg.Rect(card_x, card_y, card_w, card_h)
        pg.draw.rect(screen, (16, 22, 36), panel, border_radius=12)
        pg.draw.rect(screen, (40, 56, 92), panel, width=2, border_radius=12)
        # Title
        title = big_font.render("Selected Worm", True, (210, 220, 240))
        screen.blit(title, (panel.x + 10, panel.y + 8))
        # DNA bars
        bars = selected_worm.dna_values()
        if bars:
            bx = panel.x + 10
            by = panel.y + 34
            max_h = 50
            for i, n in enumerate(bars[:40]):  # cap to fit
                h = int(max_h * max(0.0, min(1.0, n)))
                col = DNA_COLORS[i % len(DNA_COLORS)]
                pg.draw.rect(screen, col, (bx, by + (max_h - h), 4, h), border_radius=2)
                bx += 6
        # Stats line
        info = f"S:{int(getattr(selected_worm, 'speed_ema', 0)):d}  E:{int(getattr(selected_worm, 'food_eaten', 0)):d}  Segs:{selected_worm.num_segments}"
        info_img = font.render(info, True, (230, 235, 245))
        screen.blit(info_img, (panel.x + 10, panel.y + card_h - 22))


def _draw_gl(glr: GLRenderer, maze: Maze, worms: List[Worm], cam: Tuple[float, float]) -> None:
    glr.begin(cam)
    obs_instances = []
    for obs in maze.obstacles:
        cx = obs.rect.centerx
        cy = obs.rect.centery
        rx = max(1.0, obs.rect.w * 0.5)
        ry = max(1.0, obs.rect.h * 0.5)
        obs_instances.append(CircleInstance(cx, cy, rx, ry, 34/255.0, 45/255.0, 78/255.0))
    glr.draw_instances(obs_instances)
    vw, vh = WIN_SIZE
    foods = list(maze.foods_near(cam[0] + vw * 0.5, cam[1] + vh * 0.5, max(vw, vh) * 0.8))
    food_r = float(max(1, int(maze.food_radius)))
    food_instances = [
        CircleInstance(f.x, f.y, food_r, food_r, 90/255.0, 200/255.0, 110/255.0)
        for f in foods if not f.eaten
    ]
    glr.draw_instances(food_instances)
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


def _gl_draw_selected_card(glr: GLRenderer, worm: Worm, font: pg.font.Font, big_font: pg.font.Font) -> None:
    # Build a Pygame surface for the card and upload as a texture to draw via GL
    card_w, card_h = 260, 120
    surf = pg.Surface((card_w, card_h), flags=pg.SRCALPHA)
    # Background
    pg.draw.rect(surf, (16, 22, 36), pg.Rect(0, 0, card_w, card_h), border_radius=12)
    pg.draw.rect(surf, (40, 56, 92), pg.Rect(0, 0, card_w, card_h), width=2, border_radius=12)
    # Title
    title = big_font.render("Selected Worm", True, (210, 220, 240))
    surf.blit(title, (10, 8))
    # DNA bars
    bars = worm.dna_values()
    if bars:
        from .config import DNA_COLORS
        bx = 10
        by = 34
        max_h = 50
        for i, n in enumerate(bars[:40]):
            h = int(max_h * max(0.0, min(1.0, n)))
            col = DNA_COLORS[i % len(DNA_COLORS)]
            pg.draw.rect(surf, col, (bx, by + (max_h - h), 4, h), border_radius=2)
            bx += 6
    # Stats line
    info = f"S:{int(getattr(worm, 'speed_ema', 0)):d}  E:{int(getattr(worm, 'food_eaten', 0)):d}  Segs:{worm.num_segments}"
    info_img = font.render(info, True, (230, 235, 245))
    surf.blit(info_img, (10, card_h - 22))
    # Upload and blit at top-right
    rgba = pg.image.tostring(surf, "RGBA")
    x = WIN_SIZE[0] - card_w - 12
    # Place below where minimap would be (keep consistent)
    mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
    y = 12 + mini_h + 10
    glr.blit_ui_rgba(rgba, card_w, card_h, x, y)


def _gl_draw_top_bar(glr: GLRenderer, maze: Maze, worms: List[Worm], t_sec: float, next_food_spawn_t: float,
                     births_count: int, deaths_count: int, sim_speed: float, font: pg.font.Font) -> None:
    # Compute metrics (mirror CPU HUD)
    total_food = len(maze.foods)
    available_food = maze.available_food()
    ratio = (available_food / total_food) if total_food > 0 else 0.0
    ready = sum(1 for w in worms if getattr(w, 'ready_to_mate', False))
    avg_spd = (sum(getattr(w, 'speed_ema', 0.0) for w in worms) / len(worms)) if worms else 0.0
    max_spd = max((getattr(w, 'speed_ema', 0.0) for w in worms), default=0.0)
    spawn_in = max(0, int(next_food_spawn_t - t_sec))

    # Layout matching CPU bar region (left area under top)
    mini_w = 240
    bar_x = 12
    bar_y = 8
    bar_w = max(100, (WIN_SIZE[0] - mini_w - 12 - 12) - bar_x)
    bar_h = 66
    surf = pg.Surface((bar_w, bar_h), flags=pg.SRCALPHA)
    pg.draw.rect(surf, (16, 22, 36), pg.Rect(0, 0, bar_w, bar_h), border_radius=10)
    pg.draw.rect(surf, (40, 56, 92), pg.Rect(0, 0, bar_w, bar_h), width=1, border_radius=10)
    # Row 1
    x = 10
    y = 6
    for txt in [
        f"Pop {len(worms)}",
        f"Ready {ready}",
        f"Births {births_count}",
        f"Deaths {deaths_count}",
        f"Food {available_food}/{total_food} ({ratio:.0%})",
    ]:
        img = font.render(txt, True, (220, 230, 245))
        surf.blit(img, (x, y))
        x += img.get_width() + 18
    # Row 2
    x = 10
    y = 6 + 18
    mm = int(t_sec) // 60
    ss = int(t_sec) % 60
    for txt in [
        f"Time {mm:02d}:{ss:02d}",
        f"AvgSpd {avg_spd:.0f}",
        f"MaxSpd {max_spd:.0f}",
        f"NextFood {spawn_in}s",
        f"Speed x{sim_speed:.2f}",
    ]:
        img = font.render(txt, True, (200, 215, 235))
        surf.blit(img, (x, y))
        x += img.get_width() + 18
    rgba = pg.image.tostring(surf, "RGBA")
    glr.blit_ui_rgba(rgba, bar_w, bar_h, bar_x, bar_y)


def _gl_draw_minimap(glr: GLRenderer, maze: Maze, worms: List[Worm], cam: Tuple[float, float]) -> None:
    # Create a surface for minimap and draw via existing maze API
    mini_w = 240
    mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
    surf = pg.Surface((mini_w, mini_h), flags=pg.SRCALPHA)
    dest = pg.Rect(0, 0, mini_w, mini_h)
    cam_rect = pg.Rect(int(cam[0]), int(cam[1]), WIN_SIZE[0], WIN_SIZE[1])
    positions = [w.head_pos() for w in worms]
    ready_flags = [getattr(w, 'ready_to_mate', False) for w in worms]
    maze.draw_minimap(surf, dest, positions, cam_rect, ready_flags)
    rgba = pg.image.tostring(surf, "RGBA")
    x = WIN_SIZE[0] - mini_w - 12
    y = 12
    glr.blit_ui_rgba(rgba, mini_w, mini_h, x, y)


def run() -> None:
    pg.init()
    screen, glr, use_gl = _init_window()
    clock = pg.time.Clock()

    maze = Maze(WORLD_SIZE, margin=20)
    base_params = Params()

    # Create initial worms
    random.seed()
    worms: List[Worm] = []
    for i in range(200):
        x = random.uniform(WORLD_SIZE[0] * 0.1, WORLD_SIZE[0] * 0.9)
        y = random.uniform(WORLD_SIZE[1] * 0.1, WORLD_SIZE[1] * 0.9)
        heading = random.uniform(0, 2 * math.pi)
        hue = random.uniform(0, 360)
        import colorsys as _cs
        r, g, b = _cs.hsv_to_rgb(hue / 360.0, 0.3 + random.uniform(0, 0.4), 0.85 + random.uniform(0, 0.15))
        color = (int(r * 255), int(g * 255), int(b * 255))
        worm_genetics = create_random_genetics(base_params, variation=0.5, rng=random)
        num_segments = int(getattr(worm_genetics, 'segments', 6))
        body_radius = BASE_BODY_RADIUS * random.uniform(0.6, 1.5)
        seg_spacing = BASE_SEG_SPACING * random.uniform(0.7, 1.3)
        worms.append(Worm(origin=(x, y), heading=heading, params=worm_genetics, color=color,
                          segments=num_segments, body_radius=body_radius, seg_spacing=seg_spacing))
    main_worm = worms[0]

    # Ensure enough initial food for the larger population
    from .config import FOOD_TARGET_PER_WORM
    target_avail = int(FOOD_TARGET_PER_WORM * len(worms))
    _guard = 0
    while hasattr(maze, 'available_food') and maze.available_food() < target_avail and _guard < 10000:
        maze.spawn_food_random()
        _guard += 1

    # Debug genetics
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
    slider_panel_rect = pg.Rect(WIN_SIZE[0] - 270, 110, 250, WIN_SIZE[1] - 130)
    slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)
    neuron_btn = Button(pg.Rect(WIN_SIZE[0] - 260, 70, 180, 28), "Neurons (Main Worm)", get_on=lambda: main_worm.params.neurons_enabled, on_toggle=lambda v: setattr(main_worm.params, "neurons_enabled", bool(v)))

    show_ui = False
    camera_overview = True
    manual_cam_target = np.array([WORLD_SIZE[0] * 0.5, WORLD_SIZE[1] * 0.5], dtype=float)
    camera_speed = 800.0

    t_sec = 0.0
    sim_speed = 1.0  # world speed multiplier
    cam = np.array([0.0, 0.0], dtype=float)
    def _schedule_next_food_spawn(now: float) -> float:
        return now + random.uniform(FOOD_SPAWN_MIN_S, FOOD_SPAWN_MAX_S)
    next_food_spawn_t = _schedule_next_food_spawn(t_sec)

    # Stats
    import os, csv
    from datetime import datetime
    stats_dir = os.path.join('.', 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    initial_food_total = sum(1 for f in maze.foods if not f.eaten)
    food_spawned_count = 0
    stats_log = {k: [] for k in [
        't','pop','ready','births','deaths','food_avail','food_total','food_ratio','avg_spd','max_spd'
    ]}
    globals()['_last_stats_t'] = -1.0
    births_count = 0
    deaths_count = 0

    running = True
    selected_worm: Worm | None = None
    selected_idx: int | None = None
    last_click_ms: int = -1000
    last_click_worm_id: int | None = None
    while running:
        dt_ms = clock.tick(TARGET_FPS)
        dt = dt_ms / 1000.0
        # Apply world speed multiplier to simulation time
        sim_dt = dt * sim_speed
        sim_dt_ms = max(1, int(dt_ms * sim_speed))
        t_sec += sim_dt

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif ev.key == pg.K_h:
                    show_ui = not show_ui
                elif ev.key == pg.K_v:
                    camera_overview = not camera_overview
                    if not camera_overview and selected_worm is not None:
                        main_worm = selected_worm
                elif ev.key == pg.K_r:
                    maze.regenerate()
                elif ev.key == pg.K_f:
                    if selected_worm is not None:
                        main_worm = selected_worm
                        camera_overview = False
                elif ev.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                    # Speed up (doubling for noticeable effect)
                    sim_speed = min(16.0, sim_speed * 2.0)
                elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    # Slow down (halving)
                    sim_speed = max(0.125, sim_speed / 2.0)
                elif ev.key == pg.K_0:
                    # Reset speed
                    sim_speed = 1.0
                elif ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET, pg.K_TAB):
                    if worms:
                        if selected_worm in worms:
                            selected_idx = worms.index(selected_worm)
                        elif selected_idx is None:
                            selected_idx = 0
                        step = 1
                        if ev.key == pg.K_LEFTBRACKET or (ev.key == pg.K_TAB and (pg.key.get_mods() & pg.KMOD_SHIFT)):
                            step = -1
                        selected_idx = (0 if selected_idx is None else selected_idx)
                        selected_idx = (selected_idx + step) % len(worms)
                        selected_worm = worms[selected_idx]
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                # Selection: click on a worm head in main viewport to select
                # (ignore minimap and UI area)
                mini_w = 240
                mini_h = int(240 * (WORLD_SIZE[1] / WORLD_SIZE[0]))
                mini_rect = pg.Rect(WIN_SIZE[0] - mini_w - 12, 12, mini_w, mini_h)
                in_minimap = mini_rect.collidepoint(ev.pos)
                in_ui_x = ev.pos[0] >= (WIN_SIZE[0] - 300) if show_ui else False
                did_select = False
                if not in_minimap and not in_ui_x:
                    mx, my = ev.pos
                    # Find nearest worm by head screen distance
                    best = None
                    best_d2 = float('inf')
                    for w in worms:
                        hx, hy = w.head_pos()
                        sx = hx - cam[0]
                        sy = hy - cam[1]
                        dx = sx - mx
                        dy = sy - my
                        d2 = dx*dx + dy*dy
                        pick_r = max(12.0, w.body_radius * 1.2)
                        if d2 <= (pick_r*pick_r) and d2 < best_d2:
                            best = w
                            best_d2 = d2
                    if best is not None:
                        selected_worm = best
                        if best in worms:
                            selected_idx = worms.index(best)
                        # Double-click to follow
                        now_ms = pg.time.get_ticks()
                        if last_click_worm_id == id(best) and (now_ms - last_click_ms) <= 350:
                            main_worm = best
                            camera_overview = False
                        last_click_worm_id = id(best)
                        last_click_ms = now_ms
                        did_select = True
                # Teleport only if not selecting
                if camera_overview and not did_select:
                    half_w = WIN_SIZE[0] * 0.5
                    half_h = WIN_SIZE[1] * 0.5
                    if in_minimap:
                        sx = (ev.pos[0] - mini_rect.x) / max(1, mini_rect.w)
                        sy = (ev.pos[1] - mini_rect.y) / max(1, mini_rect.h)
                        world_x = sx * WORLD_SIZE[0]
                        world_y = sy * WORLD_SIZE[1]
                        cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], world_x - half_w))
                        cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], world_y - half_h))
                        manual_cam_target = np.array([cam[0] + half_w, cam[1] + half_h], dtype=float)
                    elif not in_ui_x:
                        world_x = ev.pos[0] + cam[0]
                        world_y = ev.pos[1] + cam[1]
                        cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], world_x - half_w))
                        cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], world_y - half_h))
                        manual_cam_target = np.array([world_x, world_y], dtype=float)
            if show_ui:
                neuron_btn.handle_event(ev)
                slider_panel.handle_event(ev)

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
            manual_cam_target[0] = max(WIN_SIZE[0] * 0.5, min(WORLD_SIZE[0] - WIN_SIZE[0] * 0.5, manual_cam_target[0]))
            manual_cam_target[1] = max(WIN_SIZE[1] * 0.5, min(WORLD_SIZE[1] - WIN_SIZE[1] * 0.5, manual_cam_target[1]))

        # Update worms
        for w in worms:
            w.update(sim_dt_ms, t_sec, maze)

        # Spawn foods dynamically
        if t_sec >= next_food_spawn_t:
            available = maze.available_food()
            total_food = len(maze.foods)
            ratio = (available / total_food) if total_food > 0 else 1.0
            target = int(FOOD_TARGET_PER_WORM * len(worms))
            deficit = max(0, target - available)
            nspawn = max(1, min(FOOD_SPAWN_BURST_MAX, deficit))
            # If global availability ratio is low, boost spawn multiplier
            from .config import FOOD_LOW_RATIO, FOOD_LOW_RATIO_MULTIPLIER
            if ratio <= FOOD_LOW_RATIO:
                nspawn = max(1, min(FOOD_SPAWN_BURST_MAX * FOOD_LOW_RATIO_MULTIPLIER, deficit, nspawn * FOOD_LOW_RATIO_MULTIPLIER))
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
                sliders = build_sliders(main_worm.params, main_worm, maze, start=(WIN_SIZE[0] - 260, 120))
                slider_panel = ScrollableSliderPanel(slider_panel_rect, sliders)
            if selected_worm is not None and selected_worm not in worms:
                selected_worm = None
                selected_idx = None

        # Reproduction via spatial hash neighbor checks
        new_worms: List[Worm] = []
        grid = SpatialHash(cell_size=160.0)
        for w in worms:
            hx, hy = w.head_pos()
            grid.insert(w, hx, hy)
        visited: set[tuple[int, int]] = set()
        for wi in list(worms):
            if not wi.alive:
                continue
            hix, hiy = wi.head_pos()
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
                    offspring_n = random.randint(0, 5)
                    for _ in range(offspring_n):
                        # inherit genetic segments from donor
                        num_segments = int(getattr(child_params, 'segments', getattr(donor.params, 'segments', 6)))
                        body_radius = BASE_BODY_RADIUS * random.uniform(0.6, 1.5)
                        seg_spacing = BASE_SEG_SPACING * random.uniform(0.7, 1.3)
                        # Spawn exactly at the contact (midpoint between heads)
                        cxp = (hix + hjx) * 0.5
                        cyp = (hiy + hjy) * 0.5
                        chead = random.uniform(0, 2 * math.pi)
                        col = donor.color
                        new_worms.append(Worm(origin=(cxp, cyp), heading=chead, params=child_params, color=col,
                                              segments=num_segments, body_radius=body_radius, seg_spacing=seg_spacing))
                    # Only one parent (donor) dies after mating
                    donor.alive = False
        if new_worms:
            births_count += len(new_worms)
            worms.extend(new_worms)

        # Camera
        if camera_overview:
            cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], manual_cam_target[0] - WIN_SIZE[0] * 0.5))
            cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], manual_cam_target[1] - WIN_SIZE[1] * 0.5))
        else:
            hx, hy = main_worm.head_pos()
            cam[0] = max(0.0, min(WORLD_SIZE[0] - WIN_SIZE[0], hx - WIN_SIZE[0] * 0.5))
            cam[1] = max(0.0, min(WORLD_SIZE[1] - WIN_SIZE[1], hy - WIN_SIZE[1] * 0.5))

        # Stats sampling (~2 Hz)
        total_food = len(maze.foods)
        available_food = maze.available_food()
        ratio = (available_food / total_food) if total_food > 0 else 0.0
        ready = sum(1 for w in worms if getattr(w, 'ready_to_mate', False))
        avg_spd = (sum(getattr(w, 'speed_ema', 0.0) for w in worms) / len(worms)) if worms else 0.0
        max_spd = max((getattr(w, 'speed_ema', 0.0) for w in worms), default=0.0)
        if pg.time.get_ticks() - int(globals().get('_last_stats_ms', -1)) >= 500:
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
            globals()['_last_stats_ms'] = pg.time.get_ticks()

        # Render
        if use_gl and glr is not None:
            _draw_gl(glr, maze, worms, (cam[0], cam[1]))
            # Overlay minimap and HUD
            _gl_draw_minimap(glr, maze, worms, (cam[0], cam[1]))
            _gl_draw_top_bar(glr, maze, worms, t_sec, next_food_spawn_t, births_count, deaths_count, sim_speed, font)
            if selected_worm is not None:
                _gl_draw_selected_card(glr, selected_worm, font, big_font)
            # Swap the OpenGL buffers
            pg.display.flip()
        else:
            assert screen is not None
            _draw_cpu(screen, maze, worms, (cam[0], cam[1]), camera_overview, font, big_font, show_ui, slider_panel, neuron_btn,
                      t_sec, next_food_spawn_t, births_count, deaths_count, sim_speed, selected_worm=selected_worm)
            pg.display.flip()

    # Clean up and write stats
    try:
        pg.quit()
    finally:
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
        except Exception:
            pass
