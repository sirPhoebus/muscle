from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Any, Set


class SpatialHash:
    """Simple 2D spatial hash grid for proximity queries.

    Stores arbitrary objects keyed by their id() and a (x, y) point.
    Use for many objects with localized queries (e.g., foods, agents).
    """

    def __init__(self, cell_size: float = 100.0):
        self.cell_size = float(cell_size)
        self.cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.objects: Dict[int, Tuple[Any, float, float]] = {}

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        cs = self.cell_size
        return int(x // cs), int(y // cs)

    def clear(self) -> None:
        self.cells.clear()
        self.objects.clear()

    def insert(self, obj: Any, x: float, y: float) -> None:
        oid = id(obj)
        if oid in self.objects:
            # Update via move
            self.move(obj, x, y)
            return
        cell = self._cell(x, y)
        self.cells[cell].add(oid)
        self.objects[oid] = (obj, x, y)

    def remove(self, obj: Any) -> None:
        oid = id(obj)
        rec = self.objects.pop(oid, None)
        if rec is None:
            return
        _, x, y = rec
        cell = self._cell(x, y)
        s = self.cells.get(cell)
        if s is not None:
            s.discard(oid)
            if not s:
                self.cells.pop(cell, None)

    def move(self, obj: Any, x: float, y: float) -> None:
        oid = id(obj)
        rec = self.objects.get(oid)
        if rec is None:
            self.insert(obj, x, y)
            return
        _, ox, oy = rec
        oc = self._cell(ox, oy)
        nc = self._cell(x, y)
        if oc != nc:
            s = self.cells.get(oc)
            if s is not None:
                s.discard(oid)
                if not s:
                    self.cells.pop(oc, None)
            self.cells[nc].add(oid)
        self.objects[oid] = (obj, x, y)

    def query_radius(self, x: float, y: float, r: float) -> Iterable[Any]:
        """Return objects with positions within radius r (Euclidean)."""
        cs = self.cell_size
        cx, cy = self._cell(x, y)
        cr = int(max(0, (r // cs) + 1))
        r2 = float(r * r)
        # Iterate neighboring cells
        for gx in range(cx - cr, cx + cr + 1):
            for gy in range(cy - cr, cy + cr + 1):
                s = self.cells.get((gx, gy))
                if not s:
                    continue
                for oid in s:
                    obj, ox, oy = self.objects.get(oid, (None, 0.0, 0.0))
                    if obj is None:
                        continue
                    dx = ox - x
                    dy = oy - y
                    if (dx * dx + dy * dy) <= r2:
                        yield obj

