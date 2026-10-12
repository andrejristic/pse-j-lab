"""Daily spend cap, with honest labelling of what is and isn't a real price.

Only the Anthropic per-token prices below are actual prices. Everything else
is recorded at zero and labelled `estimated: false` -- a made-up number in a
budget report is worse than no number, because it looks like knowledge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..core import config, journal

# USD per million tokens (input, output), from the current Anthropic model table
CLAUDE_PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}


def price_for(provider: str, model: str) -> tuple | None:
    if provider != "claude":
        return None
    for k, v in CLAUDE_PRICES.items():
        if model.startswith(k):
            return v
    return None


class Budget:
    def __init__(self, path: Path | None = None, daily_usd: float | None = None):
        self.path = Path(path) if path else config.state_dir() / "budget.json"
        s = config.load_settings()
        self.daily_usd = float(daily_usd if daily_usd is not None
                               else s.get("budget_daily_usd", 2.0))
        self.data = self._load()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"days": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def spent_today(self) -> float:
        return float(self.data["days"].get(self._today(), {}).get("usd", 0.0))

    def remaining(self) -> float:
        return max(0.0, self.daily_usd - self.spent_today())

    def check(self, provider: str, model: str) -> dict:
        """Called before a request.  Free/unpriced providers are never blocked."""
        p = price_for(provider, model)
        if p is None:
            return {"allowed": True, "reason": "no per-token price known for this provider"}
        if self.remaining() <= 0:
            return {"allowed": False,
                    "reason": f"daily cap ${self.daily_usd:.2f} reached "
                              f"(spent ${self.spent_today():.4f})"}
        return {"allowed": True, "remaining_usd": self.remaining()}

    def record(self, provider: str, model: str, in_tok: int, out_tok: int,
               role: str = "") -> dict:
        p = price_for(provider, model)
        if p is None:
            usd, estimated = 0.0, False
        else:
            usd = in_tok / 1e6 * p[0] + out_tok / 1e6 * p[1]
            estimated = True
        day = self.data["days"].setdefault(self._today(),
                                           {"usd": 0.0, "calls": 0, "tokens": 0})
        day["usd"] = round(day["usd"] + usd, 6)
        day["calls"] += 1
        day["tokens"] += in_tok + out_tok
        self._save()
        rec = {"provider": provider, "model": model, "role": role,
               "input_tokens": in_tok, "output_tokens": out_tok,
               "usd": round(usd, 6), "priced": estimated,
               "spent_today": day["usd"], "remaining": self.remaining()}
        journal.log("LLM_SPEND", rec, actor="llm")
        return rec

    def report(self, days: int = 14) -> dict:
        keys = sorted(self.data["days"])[-days:]
        return {"daily_cap_usd": self.daily_usd,
                "today": self.data["days"].get(self._today(), {}),
                "history": {k: self.data["days"][k] for k in keys},
                "note": ("only Anthropic calls carry a real per-token price; "
                         "other providers are recorded at $0 and are not "
                         "an estimate of their true cost")}


_budget: Budget | None = None


def get_budget() -> Budget:
    global _budget
    if _budget is None:
        _budget = Budget()
    return _budget
