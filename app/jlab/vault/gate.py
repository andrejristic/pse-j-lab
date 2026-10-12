"""THE GATE.  One function writes cards; everything else proposes.

The rules are refusals, not warnings:

  COMPUTED without a run_id whose sha256 matches the sealed run  -> rejected
  CITED without a resolvable reference                           -> rejected
  SPECULATION carrying a run_id or reference                     -> rejected
    (if it has evidence it is not speculation; pick the right kind)
  any card whose fingerprint already exists                      -> rejected

Every rejection is journalled.  Rejections are not failures of the app, they
are the app working; a vault that never rejects anything is a notebook.
"""

from __future__ import annotations

import time

from ..core import journal, theta
from ..lab.runner import verify_run
from .cards import Card, CardKind, new_card, reference_ok
from .store import get_store


class GateError(Exception):
    pass


def _reject(reason: str, kind: str, claim: str) -> None:
    journal.log("GATE_REJECT", {"reason": reason, "kind": kind,
                                "claim": claim[:200]}, actor="gate")
    theta.bump("gate_reject", metadata={"reason": reason})
    raise GateError(reason)


def admit(kind: str, claim: str, *, run_id: str = "", sha256: str = "",
          reference: str = "", topic: str = "", body: str = "",
          tags=None, author: str = "user", lineage=None,
          verify: bool = True) -> Card:
    """The single write path into the vault."""
    kind = (kind or "").upper().strip()
    claim = (claim or "").strip()
    if kind not in CardKind.ALL:
        _reject(f"unknown card kind {kind!r}", kind, claim)
    if len(claim) < 8:
        _reject("claim too short to be a claim", kind, claim)

    if kind == CardKind.COMPUTED:
        if not run_id:
            _reject("COMPUTED needs a lab run id", kind, claim)
        if verify:
            v = verify_run(run_id)
            if not v.get("ok"):
                _reject(f"run {run_id} does not verify: {v.get('reason', 'hash mismatch')}",
                        kind, claim)
            if sha256 and sha256 != v.get("computed"):
                _reject("supplied sha256 does not match the sealed run", kind, claim)
            sha256 = v.get("computed", sha256)
    elif kind == CardKind.CITED:
        if not reference_ok(reference):
            _reject("CITED needs a resolvable reference (DOI, arXiv id, or https URL)",
                    kind, claim)
    elif kind == CardKind.SPECULATION:
        if run_id or reference:
            _reject("SPECULATION carrying evidence should be COMPUTED or CITED",
                    kind, claim)

    card = new_card(kind, claim, topic=topic, body=body, run_id=run_id,
                    sha256=sha256, reference=reference, tags=list(tags or []),
                    author=author, lineage=list(lineage or []))
    store = get_store()
    existing = store.by_fingerprint(card.fingerprint())
    if existing is not None:
        _reject(f"duplicate of card {existing.id}", kind, claim)
    store._insert(card)
    journal.log("CARD_ADMIT", {"id": card.id, "kind": kind, "topic": topic,
                               "run_id": run_id, "reference": reference,
                               "claim": claim[:200]}, actor="gate")
    return card


def propose(kind: str, claim: str, **kw) -> dict:
    """What a model is allowed to produce: a proposal, not a card.

    Returned dict says whether the gate *would* accept it and why not, so the
    GUI can show the human exactly what is missing before anything is written.
    """
    try:
        # dry run: validate without inserting
        k = (kind or "").upper().strip()
        if k not in CardKind.ALL:
            return {"ok": False, "reason": f"unknown kind {kind!r}", "proposal": kw}
        if k == CardKind.COMPUTED and not kw.get("run_id"):
            return {"ok": False, "reason": "needs a run id", "kind": k, "claim": claim}
        if k == CardKind.COMPUTED:
            v = verify_run(kw["run_id"])
            if not v.get("ok"):
                return {"ok": False, "reason": "run does not verify", "kind": k,
                        "claim": claim}
        if k == CardKind.CITED and not reference_ok(kw.get("reference", "")):
            return {"ok": False, "reason": "needs a resolvable reference",
                    "kind": k, "claim": claim}
        return {"ok": True, "kind": k, "claim": claim, "fields": kw,
                "note": "a human must admit this"}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def wound(card_id: str, attacker: str, argument: str) -> Card | None:
    """A skeptic damaged a card: it becomes SUSPECT and keeps the attack."""
    store = get_store()
    c = store.get(card_id)
    if c is None:
        raise GateError(f"no card {card_id}")
    if c.kind == CardKind.REFUTED:
        return c
    out = store._update_status(card_id, CardKind.SUSPECT,
                               f"wounded by {attacker}",
                               {"by": attacker, "argument": argument,
                                "ts": time.time(), "verdict": "suspect"})
    journal.log("CARD_WOUND", {"id": card_id, "by": attacker,
                               "argument": argument[:300]}, actor="gate")
    theta.bump("skeptic", metadata={"card": card_id})
    return out


def refute(card_id: str, attacker: str, argument: str,
           counter_run_id: str = "") -> Card | None:
    """Shown false.  The card stays forever, marked REFUTED."""
    store = get_store()
    c = store.get(card_id)
    if c is None:
        raise GateError(f"no card {card_id}")
    out = store._update_status(card_id, CardKind.REFUTED,
                               f"refuted by {attacker}",
                               {"by": attacker, "argument": argument,
                                "counter_run_id": counter_run_id,
                                "ts": time.time(), "verdict": "refuted"})
    journal.log("CARD_REFUTE", {"id": card_id, "by": attacker,
                                "counter_run_id": counter_run_id,
                                "argument": argument[:300]}, actor="gate")
    theta.bump("skeptic", metadata={"card": card_id, "refuted": True})
    return out


def card_from_run(rec, claim: str, topic: str = "", body: str = "",
                  tags=None) -> Card:
    """Convenience: turn a RunRecord straight into a COMPUTED card."""
    return admit(CardKind.COMPUTED, claim, run_id=rec.id, sha256=rec.sha256,
                 topic=topic or rec.lab, body=body,
                 tags=list(tags or []) + [rec.lab, rec.name], author="lab")
