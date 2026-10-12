"""Model access: raw REST through the Guardian, never an SDK.

An SDK would open its own sockets and bypass the egress gate, the pinning and
the journal -- and would have to be installed, which is the one thing that
cannot be relied on mid-trip.  So: urllib, four providers, one code path.
"""

from .backend import Backend, LLMError, Message, RateLimited                   # noqa: F401
from .providers import (                                          # noqa: F401
    ClaudeBackend, GeminiBackend, MistralBackend, GroqBackend,
    PROVIDERS, make_backend,
)
from .router import Router, ROLES, get_router                     # noqa: F401
from .budget import Budget, get_budget                            # noqa: F401
from .psel import system_prompt, PSEL_REGISTERS                   # noqa: F401
from .refs import verify_arxiv, verify_all_proposals              # noqa: F401
