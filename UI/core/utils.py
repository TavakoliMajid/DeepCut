from __future__ import annotations
import time
from pathlib import Path

class FPSMeter:
    def __init__(self, avg_over: int = 30):
        self.avg_over = avg_over
        self.times = []
        self.last = None
    def tick(self) -> float:
        now = time.time()
        if self.last is None:
            self.last = now
            return 0.0
        dt = now - self.last
        self.last = now
        self.times.append(dt)
        if len(self.times) > self.avg_over:
            self.times.pop(0)
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-", "+")).strip() or "class"
