"""API keys: stored 0600, never journalled, redacted from every log line."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core import config

PROVIDERS = ("claude", "gemini", "mistral", "groq")

_ENV = {
    "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}


def keys_path() -> Path:
    return config.state_dir() / "keys.json"


def _load() -> dict:
    p = keys_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save(d: dict) -> None:
    p = keys_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def set_key(provider: str, key: str) -> None:
    d = _load()
    if key:
        d[provider] = key.strip()
    else:
        d.pop(provider, None)
    _save(d)


def get_key(provider: str) -> str:
    """File first, then environment.  Missing is normal, not an error."""
    d = _load()
    if d.get(provider):
        return d[provider]
    for var in _ENV.get(provider, ()):
        v = os.environ.get(var)
        if v:
            return v.strip()
    return ""


def list_keys() -> dict:
    return {p: bool(get_key(p)) for p in PROVIDERS}


_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.I),
    re.compile(r"x-api-key:\s*\S+", re.I),
]


def redact(text: str) -> str:
    """Scrub anything key-shaped, plus the exact keys we hold."""
    if not text:
        return text
    out = str(text)
    for p in PROVIDERS:
        k = get_key(p)
        if k and len(k) > 8:
            out = out.replace(k, f"<{p}-key redacted>")
    for pat in _PATTERNS:
        out = pat.sub("<redacted>", out)
    return out
