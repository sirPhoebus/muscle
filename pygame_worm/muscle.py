from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Support running as a package or as loose scripts
try:
    from .neuro import Twitch, clamp01
except ImportError:  # pragma: no cover - fallback for direct script execution
    from neuro import Twitch, clamp01


def map_activation(a: float, baseline: float, gain: float, gamma: float) -> float:
    # Matches TS mapping: curved power law + gain and baseline
    A = clamp01(a)
    b = clamp01(baseline)
    g = max(0.0, gain)
    gm = max(0.1, gamma)
    curved = A ** gm
    toward_max = min(1.0, curved * g)
    eff = b + (1.0 - b) * toward_max
    return clamp01(eff)


@dataclass
class MuscleFiber:
    baseline: float = 0.05
    gain: float = 2.2
    gamma: float = 0.8

    twitch: Optional[Twitch] = None
    activation: float = 0.0

    def trigger(self, twitch: Twitch) -> None:
        # Overwrite current twitch with new one
        self.twitch = twitch

    def update(self) -> float:
        if self.twitch is not None:
            raw = self.twitch.step()
            if self.twitch.done():
                self.twitch = None
        else:
            raw = 0.0
        self.activation = map_activation(raw, self.baseline, self.gain, self.gamma)
        return self.activation
