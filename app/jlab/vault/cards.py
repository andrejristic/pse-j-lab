"""Card schema.  A card is small, flat, and boring on purpose."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import asdict, dataclass, field


class CardKind:
    COMPUTED = "COMPUTED"
    CITED = "CITED"
    FOLKLORE = "FOLKLORE"
    SPECULATION = "SPECULATION"
    SUSPECT = "SUSPECT"
    REFUTED = "REFUTED"
    ALL = (COMPUTED, CITED, FOLKLORE, SPECULATION, SUSPECT, REFUTED)


# a reference is resolvable if it looks like one of these
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
_ARXIV = re.compile(r"^(arXiv:)?\d{4}\.\d{4,5}(v\d+)?$", re.I)
_URL = re.compile(r"^https://\S+$")


def reference_ok(ref: str) -> bool:
    ref = (ref or "").strip()
    return bool(_DOI.match(ref) or _ARXIV.match(ref) or _URL.match(ref))


@dataclass
class Card:
    id: str
    kind: str
    claim: str
    topic: str = ""
    body: str = ""
    run_id: str = ""
    sha256: str = ""
    reference: str = ""
    tags: list = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0
    author: str = "user"
    lineage: list = field(default_factory=list)   # ids of cards this came from
    attacks: list = field(default_factory=list)   # skeptic notes
    status_note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.kind}|{self.claim.strip().lower()}|{self.run_id}|{self.reference}"
            .encode("utf-8")).hexdigest()[:16]


def new_card(kind: str, claim: str, **kw) -> Card:
    now = time.time()
    return Card(id=uuid.uuid4().hex[:12], kind=kind, claim=claim.strip(),
                created=now, updated=now, **kw)
