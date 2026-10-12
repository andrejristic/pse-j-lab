"""Guardian -- the only way this app touches the network.

Design rules, in force everywhere:

  * exactly one function performs HTTP (egress.fetch); nothing else may
  * https only, default-deny allowlist, redirects handled manually
  * TLS public-key pinning learned on first contact (so: first run at home)
  * API keys are attached only to model endpoints, never to literature sources
  * every fetched document is scanned for prompt injection before any model
    is allowed near it
  * every request, allowed or blocked, is journalled

This is a PSE unit and it has its own theta discipline: a Guardian that never
blocks anything is not vigilant, it is asleep.
"""

from .policy import ALLOWLIST, classify_host, is_allowed          # noqa: F401
from .egress import fetch, EgressError, egress_stats              # noqa: F401
from .tls import spki_sha256, pin_status, load_pins               # noqa: F401
from .content import sanitize, scan_injection, fence              # noqa: F401
from .secrets import get_key, set_key, redact, list_keys          # noqa: F401
from .unit import GuardianUnit, get_guardian                      # noqa: F401
