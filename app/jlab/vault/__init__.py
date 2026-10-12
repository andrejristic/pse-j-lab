"""The vault: every claim this app holds, and the single gate that admits them.

Card kinds and what each one costs:

  COMPUTED     must cite a run id + the sha256 of that run's result.json
  CITED        must carry a resolvable reference (DOI / arXiv id / URL)
  FOLKLORE     a claim widely repeated with no source given -- allowed, labelled
  SPECULATION  ours, unsupported; may never leave the speculation shelf
  SUSPECT      was admitted, then a skeptic wounded it
  REFUTED      shown false; kept forever, never deleted

There is exactly one write path (gate.admit).  The model never writes; it
proposes, and a proposal is not a card.
"""

from .cards import Card, CardKind, new_card                       # noqa: F401
from .gate import GateError, admit, propose, wound, refute        # noqa: F401
from .store import CardStore, get_store                           # noqa: F401
from .seed import seed_vault, TOPICS                              # noqa: F401
