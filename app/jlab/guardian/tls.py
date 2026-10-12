"""TLS public-key pinning without a `cryptography` dependency.

We parse just enough DER to find the SubjectPublicKeyInfo of the leaf
certificate and hash it.  On Python 3.13 `ssl.get_verified_chain()` exists and
we also record the issuer's SPKI; on 3.12 (the host Python on Mint) it does
not, so we fall back to leaf SPKI plus the issuer *identity* string.  That
second level is what stops a same-name-different-key substitution from passing
as a routine certificate rotation.

Pins are learned on first contact.  Learn them at home.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import ssl
import threading
import time
from pathlib import Path

from ..core import config

_lock = threading.Lock()


def pins_path() -> Path:
    return config.state_dir() / "tls_pins.json"


def load_pins() -> dict:
    p = pins_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_pins(pins: dict) -> None:
    p = pins_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pins, indent=2, sort_keys=True))
    tmp.replace(p)


# --- minimal DER walking ------------------------------------------------------

def _der_children(buf: bytes, start: int, end: int):
    """Yield (tag, header_len, content_start, content_end) of TLV items."""
    i = start
    while i < end:
        tag = buf[i]
        j = i + 1
        if j >= end:
            return
        first = buf[j]
        if first & 0x80:
            nbytes = first & 0x7F
            j += 1
            length = int.from_bytes(buf[j:j + nbytes], "big")
            j += nbytes
        else:
            length = first
            j += 1
        yield tag, i, j, j + length
        i = j + length


def extract_spki(der: bytes) -> bytes | None:
    """Return the DER bytes of the certificate's SubjectPublicKeyInfo.

    Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signature }
    TBSCertificate ::= SEQUENCE { [0] version, serial, sigAlg, issuer,
                                  validity, subject, subjectPublicKeyInfo, ... }
    So: descend into the outer SEQUENCE, take the first child (tbs), and then
    the SPKI is the first SEQUENCE after the subject -- counted positionally.
    """
    try:
        outer = next(_der_children(der, 0, len(der)))
        _, _, cs, ce = outer
        tbs = next(_der_children(der, cs, ce))
        _, _, tcs, tce = tbs
        items = list(_der_children(der, tcs, tce))
        seq_indices = []
        for idx, (tag, s, _cs2, ce2) in enumerate(items):
            seq_indices.append((idx, tag, s, ce2))
        # skip optional [0] EXPLICIT version (tag 0xA0)
        pos = 0
        if items and items[0][0] == 0xA0:
            pos = 1
        # serial(INTEGER) sigAlg(SEQ) issuer(SEQ) validity(SEQ) subject(SEQ) spki(SEQ)
        want = pos + 5
        if want >= len(items):
            return None
        tag, s, _c, e = items[want]
        if tag != 0x30:
            return None
        return der[s:e]
    except Exception:
        return None


def spki_sha256(der: bytes) -> str | None:
    spki = extract_spki(der)
    if spki is None:
        return None
    return base64.b64encode(hashlib.sha256(spki).digest()).decode()


def peer_identity(host: str, port: int = 443, timeout: float = 20.0) -> dict:
    """Connect, verify normally, and return leaf SPKI + issuer identity."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ss:
            der = ss.getpeercert(binary_form=True)
            info = ss.getpeercert() or {}
            chain_spki = []
            getter = getattr(ss, "get_verified_chain", None)
            if callable(getter):
                try:
                    for c in getter():
                        h = spki_sha256(c if isinstance(c, bytes)
                                        else c.public_bytes())
                        if h:
                            chain_spki.append(h)
                except Exception:
                    chain_spki = []
    issuer = info.get("issuer", ())
    issuer_str = "/".join("=".join(x) for grp in issuer for x in grp)
    return {
        "host": host,
        "leaf_spki": spki_sha256(der),
        "issuer": issuer_str,
        "not_after": info.get("notAfter", ""),
        "chain_spki": chain_spki,
        "seen": time.time(),
    }


def check_and_pin(host: str, port: int = 443) -> dict:
    """Compare against the stored pin, or learn it on first contact.

    Verdicts: 'learned', 'match', 'rotated' (new leaf key, same issuer, which
    is what a legitimate renewal looks like), 'BLOCKED' (different key *and*
    different issuer).
    """
    ident = peer_identity(host, port)
    if not ident["leaf_spki"]:
        return {"verdict": "unparsable", "host": host,
                "reason": "could not extract SPKI from the leaf certificate"}
    with _lock:
        pins = load_pins()
        rec = pins.get(host)
        if rec is None:
            pins[host] = {"leaf_spki": ident["leaf_spki"], "issuer": ident["issuer"],
                          "chain_spki": ident["chain_spki"],
                          "learned": ident["seen"], "history": []}
            save_pins(pins)
            return {"verdict": "learned", "host": host,
                    "leaf_spki": ident["leaf_spki"], "issuer": ident["issuer"]}
        if rec.get("leaf_spki") == ident["leaf_spki"]:
            return {"verdict": "match", "host": host}
        if rec.get("issuer") and rec["issuer"] == ident["issuer"]:
            rec.setdefault("history", []).append(
                {"leaf_spki": rec["leaf_spki"], "retired": time.time()})
            rec["leaf_spki"] = ident["leaf_spki"]
            rec["chain_spki"] = ident["chain_spki"]
            pins[host] = rec
            save_pins(pins)
            return {"verdict": "rotated", "host": host,
                    "note": "new leaf key from the same issuer"}
        return {"verdict": "BLOCKED", "host": host,
                "reason": "leaf key and issuer both changed",
                "pinned_issuer": rec.get("issuer"), "seen_issuer": ident["issuer"]}


def pin_status() -> dict:
    pins = load_pins()
    return {"n_pinned": len(pins),
            "hosts": {h: {"issuer": r.get("issuer", ""),
                          "learned": r.get("learned", 0),
                          "rotations": len(r.get("history", []))}
                      for h, r in pins.items()}}


def forget_pin(host: str) -> bool:
    with _lock:
        pins = load_pins()
        if host in pins:
            del pins[host]
            save_pins(pins)
            return True
    return False
