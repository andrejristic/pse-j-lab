"""What this app is allowed to talk to, and in what capacity.

Two classes of host, with different rights:

  MODEL       may receive an API key and our prompts.  Nothing else may.
  LITERATURE  may be read from.  Never receives a key, never receives
              anything about our work beyond a search term.

Everything else is denied.  Adding a host is a deliberate act recorded in the
journal, not a config convenience.
"""

from __future__ import annotations

MODEL_HOSTS = {
    "api.anthropic.com": "claude",
    "generativelanguage.googleapis.com": "gemini",
    "api.mistral.ai": "mistral",
    "api.groq.com": "groq",
}

LITERATURE_HOSTS = {
    "export.arxiv.org": "arXiv API",
    "arxiv.org": "arXiv",
    "ar5iv.labs.arxiv.org": "ar5iv full text",
    "api.openalex.org": "OpenAlex",
    "api.crossref.org": "Crossref",
    "api.semanticscholar.org": "Semantic Scholar",
    "en.wikipedia.org": "Wikipedia",
    "oeis.org": "OEIS",
}

ALLOWLIST = dict(MODEL_HOSTS)
ALLOWLIST.update(LITERATURE_HOSTS)

MODEL, LITERATURE, DENIED = "model", "literature", "denied"


def classify_host(host: str) -> str:
    host = (host or "").lower().strip()
    if host in MODEL_HOSTS:
        return MODEL
    if host in LITERATURE_HOSTS:
        return LITERATURE
    return DENIED


def is_allowed(host: str) -> bool:
    return classify_host(host) != DENIED


def may_carry_key(host: str) -> bool:
    """Only model endpoints ever see a credential."""
    return classify_host(host) == MODEL


# hard limits, enforced in egress
MAX_BYTES = 8 * 1024 * 1024          # a paper is not 8 MB; a zip bomb is
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 45
RATE_LIMIT_PER_HOST_PER_MINUTE = 20
