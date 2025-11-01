import math
import random
from dataclasses import dataclass


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def randn() -> float:
    # Box-Muller
    u = 0.0
    v = 0.0
    while u == 0.0:
        u = random.random()
    while v == 0.0:
        v = random.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def compute_twitch_timing(amplitude: float) -> tuple[int, int, int]:
    # Faster twitches for worm-like behavior
    amp = clamp01(amplitude)
    rise = max(30, int(90 * amp))
    hold = 20
    decay = max(120, int(520 * amp))
    return rise, hold, decay


@dataclass
class Twitch:
    target: float
    rise: int
    hold: int
    decay: int
    age: int = 0

    def value(self) -> float:
        t = self.age
        if t < self.rise:
            return (t / max(1, self.rise)) * self.target
        t -= self.rise
        if t < self.hold:
            return self.target
        t -= self.hold
        if t < self.decay:
            return self.target * (1.0 - t / max(1, self.decay))
        return 0.0

    def step(self) -> float:
        val = self.value()
        self.age += 1
        return val

    def done(self) -> bool:
        return self.age >= (self.rise + self.hold + self.decay)


class Neuron:
    """Simple neuron producing stochastic twitch amplitudes.

    Draws `sigmoid(k * N(0,1))` at `interval_ms` cadence.
    """

    def __init__(self, interval_ms: int = 300, k_gain: float = 1.3):
        self.interval_ms = max(60, int(interval_ms))
        self.k_gain = float(k_gain)
        self._accum_ms = 0
        self._pending: Twitch | None = None

    def update(self, dt_ms: int) -> Twitch | None:
        # if a twitch is pending (created last tick), return it once
        if self._pending is not None:
            tw = self._pending
            self._pending = None
            return tw

        self._accum_ms += dt_ms
        if self._accum_ms >= self.interval_ms:
            self._accum_ms -= self.interval_ms
            amp = sigmoid(self.k_gain * randn())
            rise, hold, decay = compute_twitch_timing(amp)
            self._pending = Twitch(target=amp, rise=rise, hold=hold, decay=decay)
        return None

