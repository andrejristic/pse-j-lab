"""Resolve proposed citations against arXiv before they may become cards.

This is the module that keeps a plausible-looking arXiv id out of the vault.
Every reference proposal in vault/seed.py is unverified until this code
fetches it, compares the returned title against the title we expected, and
only then hands it to the gate.

A title that comes back different from what we expected is the interesting
case: it means the id was remembered wrongly, and the proposal is rejected
rather than quietly corrected.
"""

from __future__ import annotations

import difflib
import re
import xml.etree.ElementTree as ET

from ..core import journal
from ..guardian import egress
from ..vault.cards import CardKind
from ..vault.gate import GateError, admit
from ..vault.seed import load_proposals, save_proposals

ARXIV_API = "https://export.arxiv.org/api/query?id_list={}"
_ATOM = "{http://www.w3.org/2005/Atom}"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def verify_arxiv(arxiv_id: str, expected_title: str = "") -> dict:
    """Fetch an arXiv entry and compare the real title with the expected one."""
    try:
        r = egress.fetch(ARXIV_API.format(arxiv_id), purpose="reference check")
    except Exception as e:
        return {"ok": False, "arxiv": arxiv_id, "error": f"{type(e).__name__}: {e}"}
    try:
        root = ET.fromstring(r["body"].decode("utf-8", errors="replace"))
    except ET.ParseError as e:
        return {"ok": False, "arxiv": arxiv_id, "error": f"unparsable atom feed: {e}"}
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return {"ok": False, "arxiv": arxiv_id, "error": "no entry returned"}
    title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
    if title.lower().startswith("error"):
        return {"ok": False, "arxiv": arxiv_id, "error": f"arXiv says: {title}"}
    authors = [a.findtext(f"{_ATOM}name") or "" for a in entry.findall(f"{_ATOM}author")]
    published = (entry.findtext(f"{_ATOM}published") or "")[:10]
    ratio = (difflib.SequenceMatcher(None, _norm(expected_title), _norm(title)).ratio()
             if expected_title else None)
    out = {"ok": True, "arxiv": arxiv_id, "title": title,
           "authors": authors[:8], "published": published,
           "expected_title": expected_title, "title_similarity": ratio,
           "title_matches": (ratio is None or ratio >= 0.75),
           "url": f"https://arxiv.org/abs/{arxiv_id}"}
    journal.log("REF_VERIFY", {k: out[k] for k in
                               ("arxiv", "title", "title_matches", "title_similarity")},
                actor="refs")
    return out


def verify_all_proposals(admit_cards: bool = False, progress=None) -> dict:
    """Resolve every seeded reference proposal; optionally admit the good ones."""
    rows = load_proposals()
    results, admitted, rejected = [], [], []
    for i, p in enumerate(rows):
        v = verify_arxiv(p.get("arxiv", ""), p.get("title_guess", ""))
        p["status"] = ("verified" if v.get("ok") and v.get("title_matches")
                       else ("title_mismatch" if v.get("ok") else "unresolved"))
        p["resolved_title"] = v.get("title", "")
        p["error"] = v.get("error", "")
        p["title_similarity"] = v.get("title_similarity")
        results.append({**{k: p.get(k) for k in ("key", "arxiv", "topic", "status")},
                        "resolved_title": p["resolved_title"],
                        "expected": p.get("title_guess", ""),
                        "similarity": p.get("title_similarity")})
        if admit_cards and p["status"] == "verified":
            try:
                c = admit(CardKind.CITED, p.get("claim", "") or p["resolved_title"],
                          reference=f"arXiv:{p['arxiv']}", topic=p.get("topic", ""),
                          body=f"{p['resolved_title']} ({v.get('published','')}) "
                               f"— {', '.join(v.get('authors', [])[:4])}",
                          tags=["verified-reference"], author="refs")
                admitted.append(c.id)
            except GateError as e:
                rejected.append(f"{p['arxiv']}: {e}")
        if progress:
            progress((i + 1) / max(len(rows), 1), p.get("arxiv", ""))
    save_proposals(rows)
    n_ok = sum(1 for r in results if r["status"] == "verified")
    return {"n": len(rows), "verified": n_ok,
            "mismatched": sum(1 for r in results if r["status"] == "title_mismatch"),
            "unresolved": sum(1 for r in results if r["status"] == "unresolved"),
            "admitted": admitted, "rejected": rejected, "results": results}
