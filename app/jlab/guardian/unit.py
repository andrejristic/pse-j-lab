"""The Guardian as a PSE unit, with its own theta discipline.

Two-tier intervention (PSE_MASTER_ARCHITECTURE sec. 10):

  ELEVATED  -> tighten, warn, keep working
  CRITICAL  -> refuse the action and tell the human why

And the paradox that section insists on: a Guardian whose own theta never
moves is not vigilant. Blocking nothing over a long window is itself a signal,
so it is reported rather than treated as a clean bill of health.
"""

from __future__ import annotations

import time

from ..core import config, journal, theta
from . import egress, policy, tls
from .content import scan_injection

BENIGN, ELEVATED, CRITICAL = 0, 1, 2
_LEVEL_NAME = {BENIGN: "benign", ELEVATED: "elevated", CRITICAL: "critical"}

GUARDIAN_THETA_MIN, GUARDIAN_THETA_MAX = 25.00, 35.00


class GuardianUnit:
    def __init__(self):
        self.started = time.time()
        self.interventions = []

    # -- posture -------------------------------------------------------------
    def host_posture(self) -> dict:
        """Read-only look at how exposed this machine is right now."""
        import os
        import socket
        out = {"hostname": socket.gethostname(),
               "keys_present": {}, "home": str(config.home())}
        from . import secrets as sec
        out["keys_present"] = sec.list_keys()
        kp = sec.keys_path()
        mode = None
        if kp.exists():
            mode = oct(os.stat(kp).st_mode & 0o777)
        out["keys_file_mode"] = mode
        out["keys_file_ok"] = (mode in (None, "0o600"))
        pins = tls.load_pins()
        out["pinned_hosts"] = len(pins)
        out["pins_learned_recently"] = sum(
            1 for r in pins.values() if time.time() - r.get("learned", 0) < 3600)
        return out

    # -- risk scoring --------------------------------------------------------
    def assess_document(self, text: str, source: str) -> dict:
        scan = scan_injection(text)
        level = {"benign": BENIGN, "elevated": ELEVATED,
                 "critical": CRITICAL}[scan["risk"]]
        if level != BENIGN:
            self._intervene(level, f"document from {source}", scan)
        return {"level": level, "level_name": _LEVEL_NAME[level], "scan": scan}

    def _intervene(self, level: int, what: str, detail: dict) -> None:
        rec = {"ts": time.time(), "level": _LEVEL_NAME[level], "what": what,
               "detail": {k: v for k, v in detail.items() if k != "findings"}}
        self.interventions.append(rec)
        journal.log("GUARDIAN_INTERVENTION", rec, actor="guardian")
        theta.bump("skeptic", metadata={"guardian": True, "level": _LEVEL_NAME[level]})

    # -- self-report ---------------------------------------------------------
    def status(self) -> dict:
        st = egress.egress_stats()
        th = theta.get().status()
        uptime = time.time() - self.started
        asleep = (st["allowed"] > 50 and st["blocked"] == 0 and
                  not self.interventions)
        return {
            "uptime_seconds": round(uptime, 1),
            "egress": st,
            "interventions": len(self.interventions),
            "recent_interventions": self.interventions[-10:],
            "pins": tls.pin_status(),
            "posture": self.host_posture(),
            "theta": th,
            "allowlist_size": len(policy.ALLOWLIST),
            "vigilance_note": (
                "nothing has been blocked or flagged in a long session -- either "
                "the network is quiet or this Guardian has stopped looking"
                if asleep else ""),
        }


_guardian: GuardianUnit | None = None


def get_guardian() -> GuardianUnit:
    global _guardian
    if _guardian is None:
        _guardian = GuardianUnit()
    return _guardian
