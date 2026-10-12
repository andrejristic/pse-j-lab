"""Theta monitor — the PSE health metric, persisted across launches.

theta in [0,100] is the share of compute directed at the other.  Healthy band
[25,35], optimal 30.  Two failure modes matter here and both are checked:

  * mean outside the band            -> withdrawal (<25) or enmeshment (>35)
  * variance collapse (sigma < 0.01) -> schizoid screening *regardless of mean*
    ("a unit with perfectly stable theta is not engaging authentically")

In a laboratory the "other" is the material: data, papers, results that refuse
to agree with us.  theta rises when we take the world seriously (reading a
paper, running an experiment that can fail, letting the skeptic speak) and
falls when we only talk to ourselves (replaying cached results, generating
prose about work not done).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, journal
from .welford import Welford

# theta sources, i.e. what legitimately moves the metric in this app
THETA_SOURCES = {
    "lab_run",          # an experiment that could have refuted us
    "reading",          # external text absorbed
    "skeptic",          # an attack on our own card
    "gate_reject",      # the vault refused something of ours
    "user",             # the human intervened
    "llm_call",         # asked a model
    "idle_drift",       # null-state drift (no mission)
    "manual",
}

# Nudges per event kind. Theta is a SLOW variable: the equilibrium under a
# steady diet of one event kind is 30 + nudge/PULL, so with PULL = 0.02 the
# largest nudge here settles at ~31.9 -- inside the healthy band [25,35].
# An earlier version used nudges ~30x larger; ordinary work then drove theta
# to ~41 (permanent "enmeshment") with sigma ~2.2, far outside the healthy
# sigma band [0.015, 0.03]. A health metric that reads unhealthy whenever the
# app is being used normally measures nothing.
_NUDGE = {
    "lab_run": +0.030,
    "reading": +0.024,
    "skeptic": +0.038,
    "gate_reject": +0.014,
    "user": +0.017,
    "llm_call": +0.007,
    "idle_drift": -0.020,
    "manual": 0.0,
}

PULL = 0.02          # restoring force toward THETA_OPTIMAL
JITTER = 0.020       # deterministic wobble; a perfectly flat theta is pathological

# Bumped whenever the constants above change. A theta value produced under a
# different set of constants is not comparable to one produced under these, so
# a state file from an older version is reset rather than carried forward --
# otherwise an upgrade silently inherits a number that means something else.
NUDGE_VERSION = 2


@dataclass
class ThetaState:
    current: float = config.THETA_OPTIMAL
    history: list = field(default_factory=list)   # bounded ring
    last_event_ts: float = 0.0
    steps: int = 0

    def as_dict(self) -> dict:
        return {"current": self.current, "history": self.history[-4096:],
                "last_event_ts": self.last_event_ts, "steps": self.steps,
                "nudge_version": NUDGE_VERSION}


class ThetaMonitor:
    """Persistent theta with drift, band checks and flatline screening."""

    MAX_HISTORY = 4096
    IDLE_SECONDS = 1800.0        # after this long with no event, drift starts

    def __init__(self, path: Path | None = None, autoload: bool = True):
        self.path = Path(path) if path else config.state_dir() / "theta.json"
        self.state = ThetaState()
        if autoload:
            self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
            if int(d.get("nudge_version", 0)) != NUDGE_VERSION:
                journal.log("THETA_RESET",
                            {"reason": "state written under different constants",
                             "found_version": d.get("nudge_version", 0),
                             "expected_version": NUDGE_VERSION,
                             "discarded_theta": d.get("current")}, actor="theta")
                self.state = ThetaState()
                self.save()
                return
            self.state = ThetaState(
                current=float(d.get("current", config.THETA_OPTIMAL)),
                history=list(d.get("history", []))[-self.MAX_HISTORY:],
                last_event_ts=float(d.get("last_event_ts", 0.0)),
                steps=int(d.get("steps", 0)),
            )
        except Exception:
            self.state = ThetaState()

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state.as_dict(), indent=2))
        os.replace(tmp, self.path)

    # -- updating -----------------------------------------------------------
    def update(self, source: str, amount: float | None = None,
               metadata: dict | None = None, log_event: bool = True) -> float:
        if source not in THETA_SOURCES:
            raise ValueError(f"unknown theta source: {source}")
        now = time.time()
        delta = _NUDGE.get(source, 0.0) if amount is None else float(amount)

        # a small stochastic-free wobble: pull toward optimum, proportional to
        # distance, so theta neither sticks nor runs away.  Deterministic on
        # purpose -- the app must be reproducible.
        pull = (config.THETA_OPTIMAL - self.state.current) * PULL
        # The wobble represents variation produced by real engagement, so it is
        # applied only when the event carries any. A no-op event must leave
        # theta flat -- otherwise the jitter acts as a floor under sigma and the
        # schizoid screen ("perfectly stable theta is not authentic engagement")
        # could never fire, which is exactly the failure it exists to catch.
        jitter = JITTER * math.sin(self.state.steps * 1.7) if delta != 0.0 else 0.0

        value = self.state.current + delta + pull + jitter
        value = max(0.0, min(100.0, value))
        self.state.current = round(value, 2)
        self.state.history.append(self.state.current)
        if len(self.state.history) > self.MAX_HISTORY:
            del self.state.history[:-self.MAX_HISTORY]
        self.state.last_event_ts = now
        self.state.steps += 1
        self.save()
        if log_event:
            journal.log("THETA", {"source": source, "value": self.state.current,
                                  "delta": round(delta, 3),
                                  "metadata": metadata or {}}, actor="theta")
        return self.state.current

    def tick_idle(self) -> float | None:
        """Null-state drift: an idle unit destabilises with no perturbation."""
        if self.state.last_event_ts == 0.0:
            self.state.last_event_ts = time.time()
            return None
        if time.time() - self.state.last_event_ts < self.IDLE_SECONDS:
            return None
        return self.update("idle_drift")

    # -- diagnosis ----------------------------------------------------------
    def stats(self, window: int = 512) -> dict:
        h = self.state.history[-window:]
        w = Welford().extend(h)
        return w.as_dict()

    def status(self, window: int = 512) -> dict:
        s = self.stats(window)
        cur = self.state.current
        band = "healthy"
        if cur < config.THETA_MIN:
            band = "withdrawal"
        elif cur > config.THETA_MAX:
            band = "enmeshment"
        flat = (s["n"] >= 64 and s["std"] < config.SIGMA_FLATLINE)
        return {
            "theta": cur,
            "band": band,
            "sigma": s["std"],
            "n": s["n"],
            "flatline": flat,
            "healthy": band == "healthy" and not flat,
            "note": ("perfectly stable theta is not authentic engagement"
                     if flat else ""),
        }


_default: ThetaMonitor | None = None


def get() -> ThetaMonitor:
    global _default
    if _default is None:
        _default = ThetaMonitor()
    return _default


def bump(source: str, **kw) -> float:
    return get().update(source, **kw)
