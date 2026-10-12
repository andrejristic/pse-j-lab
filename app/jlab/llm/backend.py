"""Common backend shape."""

from __future__ import annotations

from dataclasses import dataclass


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    """The provider refused for quota reasons, not because the request was bad.

    Carries enough to act on: which provider, how long it says to wait, and
    whether the quota is a *zero* allowance (a model this account simply may
    not use for free) rather than an exhausted one -- those need opposite
    responses, and the provider's own message does not distinguish them.
    """

    def __init__(self, message: str, provider: str = "", model: str = "",
                 retry_after: float | None = None, hard_zero: bool = False):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.hard_zero = hard_zero


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    raw: dict = None

    def as_dict(self) -> dict:
        return {"text": self.text, "provider": self.provider, "model": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "stop_reason": self.stop_reason}


class Backend:
    name = "abstract"
    default_model = ""

    def __init__(self, model: str = "", max_tokens: int = 1200,
                 temperature: float = 0.2):
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, messages, system: str = "") -> Completion:
        raise NotImplementedError

    def ping(self) -> dict:
        """Cheapest possible round trip, used by the first-run wizard."""
        try:
            c = self.complete([Message("user", "Reply with the single word: ok")],
                              system="Answer with one word.")
            return {"ok": True, "provider": self.name, "model": self.model,
                    "reply": c.text.strip()[:40]}
        except Exception as e:
            return {"ok": False, "provider": self.name, "model": self.model,
                    "error": f"{type(e).__name__}: {e}"}
