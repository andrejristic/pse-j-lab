"""Route by role, and make sure the skeptic is not the same model as the author.

A model asked to critique its own output agrees with itself. So SKEPTIC is
routed to a *different provider* than MATH whenever two are configured, and
the router says so in its explanation rather than doing it silently.
"""

from __future__ import annotations

from ..core import config, journal, theta
from ..guardian import secrets
from .backend import LLMError, Message, RateLimited
from .budget import get_budget
from .providers import PROVIDERS, make_backend
from .psel import scrub, system_prompt

ROLES = ("MATH", "SKEPTIC", "TRIAGE", "SUMMARIZE", "CURATE", "READING", "CHAT")

# preference order per role; first configured provider wins
_PREFER = {
    "MATH": ["claude", "gemini", "mistral", "groq"],
    "SKEPTIC": ["gemini", "mistral", "claude", "groq"],
    "TRIAGE": ["groq", "gemini", "mistral", "claude"],
    "SUMMARIZE": ["groq", "gemini", "mistral", "claude"],
    "CURATE": ["gemini", "groq", "mistral", "claude"],
    "READING": ["gemini", "claude", "mistral", "groq"],
    "CHAT": ["claude", "gemini", "mistral", "groq"],
}


class Router:
    def __init__(self):
        self.settings = config.load_settings()

    def available(self) -> list:
        return [p for p in PROVIDERS if secrets.get_key(p)]

    def pick(self, role: str) -> dict:
        role = role.upper()
        if role not in ROLES:
            role = "CHAT"
        have = self.available()
        if not have:
            return {"provider": None, "reason": "no API key configured for any provider"}
        forced = (self.settings.get("roles") or {}).get(role)
        if forced and forced in have:
            return {"provider": forced, "reason": "pinned in settings"}
        order = _PREFER[role]
        chosen = next((p for p in order if p in have), have[0])
        reason = "preference order"
        if role == "SKEPTIC" and len(have) > 1:
            math_choice = self.pick("MATH")["provider"]
            if chosen == math_choice:
                alt = next((p for p in order if p in have and p != math_choice), None)
                if alt:
                    chosen, reason = alt, ("different provider from MATH so the "
                                           "skeptic is not grading its own work")
        return {"provider": chosen, "reason": reason}

    def model_for(self, provider: str) -> str:
        return ((self.settings.get("providers") or {}).get(provider) or {}).get("model", "")

    def ask(self, role: str, prompt: str, history=None, register: str = "formal",
            max_tokens: int = 1200) -> dict:
        """One call, budgeted, journalled, scrubbed.

        If the chosen provider refuses on quota, the remaining configured
        providers are tried in preference order. A trip is not the moment to
        discover that one provider's daily allowance ends the conversation.
        """
        sel = self.pick(role)
        if not sel["provider"]:
            raise LLMError(sel["reason"])

        order = [sel["provider"]] + [p for p in _PREFER[role.upper() if role.upper()
                                                        in ROLES else "CHAT"]
                                     if p in self.available() and p != sel["provider"]]
        msgs = list(history or []) + [Message("user", prompt)]
        sys = system_prompt(role, register=register,
                            lang=self.settings.get("lang", "sr"))
        budget = get_budget()
        attempts = []
        c = None
        for i, provider in enumerate(order):
            backend = make_backend(provider, model=self.model_for(provider),
                                   max_tokens=max_tokens)
            gate = budget.check(provider, backend.model)
            if not gate["allowed"]:
                attempts.append(f"{provider}: {gate['reason']}")
                continue
            try:
                c = backend.complete(msgs, system=sys)
                if i > 0:
                    sel = {"provider": provider,
                           "reason": f"fell back after: {'; '.join(attempts)}"}
                break
            except RateLimited as e:
                attempts.append(str(e))
                journal.log("LLM_RATE_LIMITED",
                            {"provider": provider, "model": backend.model,
                             "hard_zero": e.hard_zero,
                             "retry_after": e.retry_after}, actor="llm")
                continue
        if c is None:
            raise LLMError("every configured provider refused:\n  - "
                           + "\n  - ".join(attempts or ["no provider available"]))
        provider = sel["provider"]
        spend = budget.record(provider, c.model, c.input_tokens, c.output_tokens, role)
        check = scrub(c.text)
        journal.log("LLM_CALL", {"role": role, "provider": provider, "model": c.model,
                                 "routing_reason": sel["reason"],
                                 "psel_clean": check["clean"],
                                 "psel_findings": check["n"],
                                 "chars": len(c.text)}, actor="llm")
        theta.bump("llm_call", metadata={"role": role, "provider": provider})
        return {"text": c.text, "provider": provider, "model": c.model,
                "role": role, "routing_reason": sel["reason"],
                "usage": {"input": c.input_tokens, "output": c.output_tokens},
                "spend": spend, "psel": check}

    def ping_all(self) -> dict:
        out = {}
        for p in PROVIDERS:
            if not secrets.get_key(p):
                out[p] = {"ok": False, "error": "no key configured"}
                continue
            out[p] = make_backend(p, model=self.model_for(p), max_tokens=16).ping()
        return out

    def status(self) -> dict:
        have = self.available()
        return {"configured": have,
                "roles": {r: self.pick(r) for r in ROLES},
                "budget": get_budget().report(),
                "offline": not have}


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router
