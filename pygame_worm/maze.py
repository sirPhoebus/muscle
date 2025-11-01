from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple
import math

import pygame as pg


Color = Tuple[int, int, int]


@dataclass
class EllipseObstacle:
    rect: pg.Rect  # Bounding rectangle for the ellipse

    @property
    def center(self) -> Tuple[float, float]:
        return (self.rect.centerx, self.rect.centery)

    @property
    def rx(self) -> float:
        """Horizontal radius"""
        return self.rect.w / 2.0

    @property
    def ry(self) -> float:
        """Vertical radius"""
        return self.rect.h / 2.0

    def draw(self, surf: pg.Surface, color: Color) -> None:
        pg.draw.ellipse(surf, color, self.rect)

    def collide_point(self, p: Tuple[float, float]) -> bool:
        # Check if point is inside ellipse using standard equation
        px, py = p
        cx, cy = self.center
        dx = px - cx
        dy = py - cy
        # Ellipse equation: (dx/rx)^2 + (dy/ry)^2 <= 1
        return (dx * dx) / (self.rx * self.rx) + (dy * dy) / (self.ry * self.ry) <= 1.0

    def collide_circle(self, center: Tuple[float, float], radius: float) -> bool:
        # Check if circle collides with ellipse
        cx, cy = center
        ex, ey = self.center
        # Find closest point on ellipse to circle center (approximation)
        # For simplicity, check if circle center is within ellipse expanded by radius
        dx = cx - ex
        dy = cy - ey
        # Expand ellipse by radius
        rx_expanded = self.rx + radius
        ry_expanded = self.ry + radius
        return (dx * dx) / (rx_expanded * rx_expanded) + (dy * dy) / (ry_expanded * ry_expanded) <= 1.0

    def ray_intersect(self, origin: Tuple[float, float], end: Tuple[float, float]) -> float | None:
        # Ray-ellipse intersection (simplified)
        ox, oy = origin
        ex, ey = end
        cx, cy = self.center
        
        # Direction vector
        dx = ex - ox
        dy = ey - oy
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return None
        dx /= length
        dy /= length
        
        # Transform to ellipse space (translate to center)
        px = ox - cx
        py = oy - cy
        
        # Ellipse equation: (x/rx)^2 + (y/ry)^2 = 1
        # Ray: p + t*d
        # Substitute and solve quadratic
        rx = self.rx
        ry = self.ry
        
        a = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
        b = 2.0 * ((px * dx) / (rx * rx) + (py * dy) / (ry * ry))
        c = (px * px) / (rx * rx) + (py * py) / (ry * ry) - 1.0
        
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        # Find smallest positive t within segment length
        t = None
        if 0 <= t1 <= length:
            t = t1
        if 0 <= t2 <= length and (t is None or t2 < t):
            t = t2
        
        return t if t is not None else None


@dataclass
class Food:
    x: float
    y: float
    eaten: bool = False

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


class Maze:
    def __init__(self, size: Tuple[int, int], margin: int = 20):
        self.size = size
        self.margin = margin
        self.obstacles: List[EllipseObstacle] = []
        # Food generation/render params (configurable)
        self.foods_per_cell: int = 2
        self.food_prob: float = 0.85
        self.food_radius: int = 3
        self.foods: List[Food] = []
        self.bounds = pg.Rect(0, 0, size[0], size[1])
        self._generate_default()

    def spawn_food_random(self, tries: int = 20) -> bool:
        """Spawn a single food at a random free location. Returns True if placed."""
        rnd = random.Random()
        m = self.margin
        w, h = self.size
        for _ in range(max(1, tries)):
            fx = rnd.uniform(m + 10, w - m - 10)
            fy = rnd.uniform(m + 10, h - m - 10)
            # avoid placing inside obstacles
            blocked = False
            for obs in self.obstacles:
                if obs.rect.collidepoint((fx, fy)):
                    blocked = True
                    break
            if blocked:
                continue
            self.foods.append(Food(float(fx), float(fy)))
            return True
        return False

    def _generate_default(self) -> None:
        w, h = self.size
        m = self.margin
        self.obstacles = []
        # Add more blocks with a grid-based generator scaled to world size
        # Use non-deterministic RNG so regenerate ('R') creates a new maze each time
        rnd = random.Random()
        # Choose grid resolution based on world size for a maze-like density
        target_cell = 400  # px per cell (roughly)
        cols = max(8, min(80, (w - 2 * m) // target_cell))
        rows = max(6, min(80, (h - 2 * m) // target_cell))
        cols = int(max(1, cols))
        rows = int(max(1, rows))
        cellw = (w - 2 * m) // max(1, cols)
        cellh = (h - 2 * m) // max(1, rows)

        # Per-cell blocks
        p_block = 0.45
        for r in range(rows):
            for c in range(cols):
                if rnd.random() < p_block:
                    rx = m + c * cellw + rnd.randint(6, max(7, int(0.25 * cellw)))
                    ry = m + r * cellh + rnd.randint(6, max(7, int(0.25 * cellh)))
                    rw = rnd.randint(int(0.35 * cellw), max(int(0.36 * cellw), int(0.85 * cellw)))
                    rh = rnd.randint(int(0.25 * cellh), max(int(0.26 * cellh), int(0.75 * cellh)))
                    self.obstacles.append(EllipseObstacle(pg.Rect(rx, ry, rw, rh)))

        # Add corridor-like walls (long thin rectangles) to create structure
        t_h = max(12, int(0.08 * cellh))  # horizontal wall thickness
        t_v = max(12, int(0.08 * cellw))  # vertical wall thickness
        # Horizontal runs
        for r in range(rows):
            if rnd.random() < 0.35:
                c0 = rnd.randint(0, max(0, cols - 3))
                run = rnd.randint(2, min(6, cols - c0))
                y = m + r * cellh + rnd.randint(int(0.2 * cellh), int(0.8 * cellh))
                x = m + c0 * cellw + 4
                wlen = run * cellw - 8
                self.obstacles.append(EllipseObstacle(pg.Rect(x, y, wlen, t_h)))
        # Vertical runs
        for c in range(cols):
            if rnd.random() < 0.35:
                r0 = rnd.randint(0, max(0, rows - 3))
                run = rnd.randint(2, min(6, rows - r0))
                x = m + c * cellw + rnd.randint(int(0.2 * cellw), int(0.8 * cellw))
                y = m + r0 * cellh + 4
                hlen = run * cellh - 8
                self.obstacles.append(EllipseObstacle(pg.Rect(x, y, t_v, hlen)))

        # Scatter food items roughly foods_per_cell per grid cell on average
        self.foods = []
        foods_per_cell = max(0, int(self.foods_per_cell))
        for r in range(rows):
            for c in range(cols):
                for _ in range(foods_per_cell):
                    if rnd.random() < max(0.0, min(1.0, float(self.food_prob))):
                        # Try a few times to avoid placing food inside obstacles
                        placed = False
                        for _try in range(6):
                            fx = m + c * cellw + rnd.randint(10, max(11, cellw - 10))
                            fy = m + r * cellh + rnd.randint(10, max(11, cellh - 10))
                            blocked = False
                            for obs in self.obstacles:
                                if obs.rect.collidepoint((fx, fy)):
                                    blocked = True
                                    break
                            if not blocked:
                                self.foods.append(Food(float(fx), float(fy)))
                                placed = True
                                break

    def regenerate(self) -> None:
        self._generate_default()

    def draw(self, surf: pg.Surface, cam: Tuple[float, float]) -> None:
        # Background for current viewport
        surf.fill((9, 12, 24))
        cx, cy = cam
        # Walls/blocks with camera offset - draw as ellipses
        for obs in self.obstacles:
            r = obs.rect.move(-int(cx), -int(cy))
            pg.draw.ellipse(surf, (34, 45, 78), r)
        # Border (world bounds) offset by camera
        border = self.bounds.move(-int(cx), -int(cy))
        pg.draw.rect(surf, (54, 64, 104), border, width=2)
        # Foods within viewport
        vw = surf.get_width()
        vh = surf.get_height()
        for f in self.foods:
            if f.eaten:
                continue
            sx = int(f.x - cx)
            sy = int(f.y - cy)
            if 0 <= sx < vw and 0 <= sy < vh:
                pg.draw.circle(surf, (90, 200, 110), (sx, sy), max(1, int(self.food_radius)))

    def draw_minimap(self, surf: pg.Surface, dest: pg.Rect, worm_positions: List[Tuple[float, float]], cam_rect: pg.Rect | None = None) -> None:
        # Draw a scaled overview of world
        pg.draw.rect(surf, (12, 16, 28), dest)
        pg.draw.rect(surf, (40, 50, 80), dest, width=1)
        sx = dest.w / max(1, self.size[0])
        sy = dest.h / max(1, self.size[1])
        # Obstacles - draw as ellipses in minimap too
        for obs in self.obstacles:
            rx = dest.x + int(obs.rect.x * sx)
            ry = dest.y + int(obs.rect.y * sy)
            rw = max(1, int(obs.rect.w * sx))
            rh = max(1, int(obs.rect.h * sy))
            pg.draw.ellipse(surf, (30, 42, 70), pg.Rect(rx, ry, rw, rh))
        # Foods
        for f in self.foods:
            if f.eaten:
                continue
            px = dest.x + int(f.x * sx)
            py = dest.y + int(f.y * sy)
            surf.fill((80, 180, 100), (px, py, 2, 2))
        # Worms as white dots
        for wp in worm_positions:
            wx = dest.x + int(wp[0] * sx)
            wy = dest.y + int(wp[1] * sy)
            pg.draw.circle(surf, (240, 250, 255), (wx, wy), 3)
        # Camera rect
        if cam_rect is not None:
            cr = pg.Rect(
                dest.x + int(cam_rect.x * sx),
                dest.y + int(cam_rect.y * sy),
                max(1, int(cam_rect.w * sx)),
                max(1, int(cam_rect.h * sy)),
            )
            pg.draw.rect(surf, (120, 130, 170), cr, width=1)

    def collide_circle(self, center: Tuple[float, float], radius: float) -> bool:
        # Check bounds
        x, y = center
        if x - radius < self.margin or x + radius > self.size[0] - self.margin:
            return True
        if y - radius < self.margin or y + radius > self.size[1] - self.margin:
            return True
        # Check obstacles
        for obs in self.obstacles:
            if obs.collide_circle(center, radius):
                return True
        return False

    def raycast(self, origin: Tuple[float, float], angle: float, max_dist: float) -> float:
        ox, oy = origin
        ex = ox + max_dist * math.cos(angle)
        ey = oy + max_dist * math.sin(angle)
        # clamp to bounds
        ex = max(0, min(self.size[0], ex))
        ey = max(0, min(self.size[1], ey))
        # check obstacles, keep nearest
        best = None
        # border as walls: approximate by returning distance to end if out of bounds
        base_dist = ((ex - ox) ** 2 + (ey - oy) ** 2) ** 0.5
        best = base_dist
        for obs in self.obstacles:
            d = obs.ray_intersect((ox, oy), (ex, ey))
            if d is not None:
                if best is None or d < best:
                    best = d
        return float(best if best is not None else max_dist)
