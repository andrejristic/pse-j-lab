"""Four providers, one code path: raw REST through the Guardian's egress gate.

Why not the official SDKs: every byte this app sends must pass through
guardian.egress.fetch so the allowlist, the TLS pinning, the rate limit and
the journal apply.  An SDK opens its own sockets and would bypass all four --
and it would have to be installed, which is the one thing that cannot be
relied on in the middle of a trip.

Model ids and pricing are taken from the current Anthropic model table, not
from memory.  Note the Claude 5 family rejects `temperature` / `top_p` /
`top_k` with a 400, and thinking is adaptive by default -- so no sampling
parameters are sent and no `thinking` field is set.
"""

from __future__ import annotations

import json
import re

from ..guardian import egress
from .backend import Backend, Completion, LLMError, Message, RateLimited


def _quota_detail(text: str) -> dict:
    """Pull the actionable numbers out of a provider's quota complaint.

    Observed on a real free Gemini key: the body reports
    `limit: 0` for the free-tier metrics AND `Please retry in 55s` in the same
    message, and the same model then succeeds minutes later. So `limit: 0` is
    NOT a reliable statement that the account may never use the model -- do not
    tell the user to go change models on the strength of it. Report both
    numbers and let the retry decide.
    """
    limits = [int(x) for x in re.findall(r"limit:\s*(\d+)", text)]
    wait = re.findall(r"retry in ([\d.]+)s", text)
    return {"limits": limits,
            "zero_limit_reported": bool(limits) and max(limits) == 0,
            "retry_after": float(wait[0]) if wait else None}


class ClaudeBackend(Backend):
    name = "claude"
    default_model = "claude-opus-5"
    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def complete(self, messages, system: str = "") -> Completion:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            body["system"] = system
        # deliberately no temperature/top_p/top_k: rejected by the Claude 5 family
        r = egress.fetch(self.ENDPOINT, method="POST", provider="claude",
                         headers={"Content-Type": "application/json"},
                         body=json.dumps(body).encode(), purpose="llm")
        data = json.loads(r["body"].decode("utf-8", errors="replace"))
        if r["status"] >= 400:
            raise LLMError(f"claude {r['status']}: {data.get('error', {}).get('message', '')}")
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        u = data.get("usage", {})
        return Completion(text=text, provider=self.name, model=data.get("model", self.model),
                          input_tokens=u.get("input_tokens", 0),
                          output_tokens=u.get("output_tokens", 0),
                          stop_reason=data.get("stop_reason", ""), raw=data)


class GeminiBackend(Backend):
    name = "gemini"
    # gemini-2.5-flash, not gemini-2.0-flash: on a real free key 2.0-flash was
    # rate-limited on consecutive attempts while 2.5-flash answered every time.
    # (2.0-flash does work again after a pause -- it is throttled, not barred.)
    default_model = "gemini-2.5-flash"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def complete(self, messages, system: str = "") -> Completion:
        contents = [{"role": ("model" if m.role == "assistant" else "user"),
                     "parts": [{"text": m.content}]} for m in messages]
        body = {"contents": contents,
                "generationConfig": {"maxOutputTokens": self.max_tokens,
                                     "temperature": self.temperature}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.BASE}/{self.model}:generateContent"
        r = egress.fetch(url, method="POST", provider="gemini",
                         headers={"Content-Type": "application/json"},
                         body=json.dumps(body).encode(), purpose="llm")
        raw = r["body"].decode("utf-8", errors="replace")
        data = json.loads(raw)
        if r["status"] == 429:
            q = _quota_detail(raw)
            wait = q["retry_after"]
            msg = f"gemini: free-tier quota hit for '{self.model}'"
            if wait:
                msg += f"; Google says retry in {wait:.0f}s"
            if q["zero_limit_reported"]:
                msg += (" (it also reports limit 0 for the free-tier metrics; "
                        "in practice this often clears on retry — if it does "
                        "not, switch model in the wizard)")
            raise RateLimited(msg, provider="gemini", model=self.model,
                              retry_after=wait,
                              hard_zero=q["zero_limit_reported"])
        if r["status"] >= 400:
            raise LLMError(f"gemini {r['status']}: "
                           f"{data.get('error', {}).get('message', '')[:300]}")
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback", {})
            raise LLMError(f"gemini returned no candidates"
                           + (f" (blocked: {fb.get('blockReason')})" if fb else ""))
        parts = (cands[0].get("content") or {}).get("parts")
        if not parts:
            # A thinking model can spend the whole output budget on reasoning and
            # return content with no parts at all. Silently yielding "" here made
            # the app look broken with no explanation.
            fr = cands[0].get("finishReason", "?")
            u = data.get("usageMetadata", {})
            hint = ""
            if fr == "MAX_TOKENS":
                think = u.get("thoughtsTokenCount", 0)
                hint = (f" — max_tokens={self.max_tokens} was used up before any "
                        f"answer was written")
                hint += (f", {think} of it on internal reasoning. " if think
                         else ". ")
                hint += ("Raise max_tokens, or use gemini-flash-lite-latest, "
                         "which does no internal reasoning.")
            elif fr == "SAFETY":
                hint = " — the response was blocked by a safety filter."
            raise LLMError(f"gemini returned no text (finishReason={fr}){hint}")
        text = "".join(p.get("text", "") for p in parts)
        u = data.get("usageMetadata", {})
        return Completion(text=text, provider=self.name, model=self.model,
                          input_tokens=u.get("promptTokenCount", 0),
                          output_tokens=u.get("candidatesTokenCount", 0),
                          stop_reason=(cands[0].get("finishReason", "") if cands else ""),
                          raw=data)


class _OpenAICompat(Backend):
    """Mistral and Groq both speak the OpenAI chat-completions shape."""

    ENDPOINT = ""

    def complete(self, messages, system: str = "") -> Completion:
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        body = {"model": self.model, "messages": msgs,
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        r = egress.fetch(self.ENDPOINT, method="POST", provider=self.name,
                         headers={"Content-Type": "application/json"},
                         body=json.dumps(body).encode(), purpose="llm")
        raw = r["body"].decode("utf-8", errors="replace")
        data = json.loads(raw)
        if r["status"] >= 400:
            msg = data.get("error", {})
            msg = msg.get("message", "") if isinstance(msg, dict) else str(msg)
            if r["status"] == 429:
                q = _quota_detail(raw)
                raise RateLimited(f"{self.name}: quota exhausted for "
                                  f"'{self.model}'"
                                  + (f"; retry in {q['retry_after']:.0f}s"
                                     if q["retry_after"] else ""),
                                  provider=self.name, model=self.model,
                                  retry_after=q["retry_after"],
                                  hard_zero=q["zero_limit_reported"])
            raise LLMError(f"{self.name} {r['status']}: {msg[:300]}")
        ch = (data.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content", "") or ""
        u = data.get("usage", {})
        return Completion(text=text, provider=self.name,
                          model=data.get("model", self.model),
                          input_tokens=u.get("prompt_tokens", 0),
                          output_tokens=u.get("completion_tokens", 0),
                          stop_reason=ch.get("finish_reason", ""), raw=data)


class MistralBackend(_OpenAICompat):
    name = "mistral"
    default_model = "mistral-large-latest"
    ENDPOINT = "https://api.mistral.ai/v1/chat/completions"


class GroqBackend(_OpenAICompat):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


PROVIDERS = {
    "claude": ClaudeBackend,
    "gemini": GeminiBackend,
    "mistral": MistralBackend,
    "groq": GroqBackend,
}

# Model menus offered by the first-run wizard.  Only the Anthropic prices below
# are real per-token prices; see budget.py.
MODEL_CHOICES = {
    "claude": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
    # verified against a free Gemini key: 2.0-flash has a free-tier limit of 0
    # and is deliberately absent; flash-lite does no internal thinking, which
    # makes it the safe choice when the output budget is small
    "gemini": ["gemini-2.5-flash", "gemini-flash-latest",
               "gemini-flash-lite-latest"],
    "mistral": ["mistral-large-latest", "mistral-small-latest"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
}


def make_backend(provider: str, model: str = "", **kw) -> Backend:
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}")
    return PROVIDERS[provider](model=model, **kw)
