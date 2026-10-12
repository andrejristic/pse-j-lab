"""Streaming mean/variance (Welford) with an exact parallel merge.

Used for theta statistics and for lab metrics that must not hold the whole
sample in memory on a 4 GB machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_: float = field(default=math.inf)
    max_: float = field(default=-math.inf)

    def push(self, x: float) -> "Welford":
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if x < self.min_:
            self.min_ = x
        if x > self.max_:
            self.max_ = x
        return self

    def extend(self, xs) -> "Welford":
        for x in xs:
            self.push(x)
        return self

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def merge(self, other: "Welford") -> "Welford":
        """Exact merge of two independent accumulators (Chan et al.)."""
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean, self.m2 = other.n, other.mean, other.m2
            self.min_, self.max_ = other.min_, other.max_
            return self
        n = self.n + other.n
        d = other.mean - self.mean
        mean = self.mean + d * other.n / n
        m2 = self.m2 + other.m2 + d * d * self.n * other.n / n
        self.n, self.mean, self.m2 = n, mean, m2
        self.min_ = min(self.min_, other.min_)
        self.max_ = max(self.max_, other.max_)
        return self

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": None if self.n == 0 else self.min_,
            "max": None if self.n == 0 else self.max_,
        }
