from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int
    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)
