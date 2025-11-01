from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class BrainOutput:
    steer: float  # signed, roughly -1..1
    twitch_left: float  # 0..1
    twitch_right: float  # 0..1


class Brain:
    """
    Minimal "worm brain" that maps sensor inputs to steering and occasional
    head twitches toward food. It is not trained; weights are heuristics chosen
    to push the head toward the nearest food and away from walls.

    Inputs to update(dt_ms, dL, dR, food_ang, food_dist, ranges):
      - dL, dR: distance readings (pixels) for left/right wall sensors
      - food_ang: signed angle from head dir to nearest food (radians, -pi..pi)
      - food_dist: distance to nearest food (pixels)
      - ranges: tuple (sensor_range, food_range)
    """

    def __init__(self, interval_ms: int = 160):
        self.interval_ms = max(50, int(interval_ms))
        self._accum_ms = 0

    @staticmethod
    def _saturate(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update(
        self,
        dt_ms: int,
        dL: float,
        dR: float,
        food_ang: float,
        food_dist: float,
        sensor_range: float,
        food_range: float,
    ) -> BrainOutput:
        # Normalize inputs
        # Obstacle closeness in 0..1 (1 means very close)
        cL = 1.0 - self._saturate(dL / max(1.0, sensor_range), 0.0, 1.0)
        cR = 1.0 - self._saturate(dR / max(1.0, sensor_range), 0.0, 1.0)
        # Food angle normalized to -1..1 and distance closeness 0..1
        a = self._saturate(food_ang / math.pi, -1.0, 1.0)
        fd = self._saturate(1.0 - (food_dist / max(1.0, food_range)), 0.0, 1.0)

        # Steering: away from walls (cR - cL) and toward food angle (a)
        steer_linear = 0.6 * (cR - cL) + 1.4 * a * (0.5 + 0.5 * fd)
        # Tanh to saturate
        steer = math.tanh(steer_linear)

        # Twitch logic: periodically nudge the side that reduces |a|
        self._accum_ms += dt_ms
        tL = tR = 0.0
        if self._accum_ms >= self.interval_ms and fd > 0.05:
            self._accum_ms = 0
            # Stronger twitch when closer and angle is moderate
            amp = self._saturate(0.15 + 0.85 * (fd * (1.0 - min(1.0, abs(a)))), 0.0, 1.0)
            if a > 0.0:
                tL = amp
            elif a < 0.0:
                tR = amp

        return BrainOutput(steer=steer, twitch_left=tL, twitch_right=tR)

