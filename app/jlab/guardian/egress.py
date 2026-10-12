"""THE single HTTP function.

Nothing else in this application may open a socket.  That is not a style
preference: it is the only way the allowlist, the pinning, the rate limit and
the journal can be guaranteed to apply to every byte that leaves or enters.

Redirects are followed manually so that each hop is re-checked against the
allowlist -- an allowed host that 302s to an attacker is otherwise a hole
straight through the policy.
"""

from __future__ import annotations

import gzip
import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict, deque

from ..core import journal, theta
from . import policy, secrets, tls

USER_AGENT = "PSE-J-Lab/0.1 (research assistant; contact via app settings)"


class EgressError(Exception):
    pass


_rate = defaultdict(deque)
_rate_lock = threading.Lock()
_stats = {"allowed": 0, "blocked": 0, "errors": 0, "bytes": 0,
          "by_host": defaultdict(int), "blocks": []}


def _rate_ok(host: str) -> bool:
    now = time.time()
    with _rate_lock:
        q = _rate[host]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= policy.RATE_LIMIT_PER_HOST_PER_MINUTE:
            return False
        q.append(now)
        return True


def _block(reason: str, url: str, host: str, **extra):
    _stats["blocked"] += 1
    _stats["blocks"].append({"ts": time.time(), "host": host, "reason": reason})
    journal.log("EGRESS_BLOCK", dict({"host": host, "reason": reason,
                                      "url": secrets.redact(url)}, **extra),
                actor="guardian")
    raise EgressError(f"{reason} ({host})")


def _decompress(raw: bytes, encoding: str) -> bytes:
    enc = (encoding or "").lower()
    try:
        if "gzip" in enc:
            return gzip.decompress(raw)
        if "deflate" in enc:
            return zlib.decompress(raw)
    except Exception:
        return raw
    return raw


def fetch(url: str, *, method: str = "GET", headers: dict | None = None,
          body: bytes | None = None, provider: str | None = None,
          timeout: float | None = None, max_bytes: int | None = None,
          purpose: str = "") -> dict:
    """The only network call in the app.

    provider: name of a model provider whose key may be attached.  Attaching a
    key to a host that is not that provider's endpoint is refused.
    """
    timeout = timeout or policy.TIMEOUT_SECONDS
    max_bytes = max_bytes or policy.MAX_BYTES
    seen = []
    current = url
    for hop in range(policy.MAX_REDIRECTS + 1):
        parts = urllib.parse.urlsplit(current)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https":
            _block("only https is allowed", current, host)
        if not policy.is_allowed(host):
            _block("host not on the allowlist", current, host)
        if provider and not policy.may_carry_key(host):
            _block("refusing to send a credential to a non-model host", current, host)
        if not _rate_ok(host):
            _block("rate limit exceeded for this host", current, host)

        pin = tls.check_and_pin(host)
        if pin["verdict"] == "BLOCKED":
            _block("TLS pin mismatch", current, host, pin=pin)
        if pin["verdict"] == "unparsable":
            journal.log("EGRESS_PIN_WARN", {"host": host, "detail": pin},
                        actor="guardian")

        h = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        h.update(headers or {})
        if provider:
            key = secrets.get_key(provider)
            if not key:
                _block(f"no API key configured for {provider}", current, host)
            if provider == "claude":
                h["x-api-key"] = key
                h["anthropic-version"] = "2023-06-01"
            elif provider == "gemini":
                h["x-goog-api-key"] = key
            else:
                h["Authorization"] = f"Bearer {key}"

        req = urllib.request.Request(current, data=body, headers=h, method=method)
        opener = urllib.request.build_opener(_NoRedirect())
        t0 = time.perf_counter()
        try:
            resp = opener.open(req, timeout=timeout)
            status = resp.status
            raw = resp.read(max_bytes + 1)
            resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            status = e.code
            raw = e.read(max_bytes + 1) if e.fp else b""
            resp_headers = dict(e.headers or {})
        except Exception as e:
            _stats["errors"] += 1
            journal.log("EGRESS_ERROR", {"host": host, "error": f"{type(e).__name__}: {e}",
                                         "url": secrets.redact(current)},
                        actor="guardian")
            raise EgressError(f"{type(e).__name__}: {e}") from e

        if len(raw) > max_bytes:
            _block(f"response exceeds {max_bytes} bytes", current, host)

        if status in (301, 302, 303, 307, 308):
            loc = resp_headers.get("Location") or resp_headers.get("location")
            if not loc:
                _block("redirect without a location", current, host)
            nxt = urllib.parse.urljoin(current, loc)
            seen.append({"from": current, "to": nxt, "status": status})
            journal.log("EGRESS_REDIRECT", {"from_host": host,
                                            "to": secrets.redact(nxt),
                                            "status": status}, actor="guardian")
            current = nxt
            continue

        data = _decompress(raw, resp_headers.get("Content-Encoding", ""))
        _stats["allowed"] += 1
        _stats["bytes"] += len(data)
        _stats["by_host"][host] += 1
        journal.log("EGRESS", {"host": host, "status": status,
                               "bytes": len(data), "hops": len(seen),
                               "purpose": purpose,
                               "provider": provider or "",
                               "ms": round((time.perf_counter() - t0) * 1000, 1)},
                    actor="guardian")
        theta.bump("reading" if provider is None else "llm_call",
                   metadata={"host": host, "purpose": purpose}, )
        return {"status": status, "headers": resp_headers, "body": data,
                "host": host, "url": current, "redirects": seen,
                "pin": pin["verdict"]}

    _block("too many redirects", current, urllib.parse.urlsplit(current).hostname or "")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Disable urllib's automatic redirects; we re-check every hop ourselves."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_text(url: str, **kw) -> str:
    r = fetch(url, **kw)
    return r["body"].decode("utf-8", errors="replace")


def fetch_json(url: str, **kw) -> dict:
    r = fetch(url, **kw)
    return json.loads(r["body"].decode("utf-8", errors="replace"))


def egress_stats() -> dict:
    return {"allowed": _stats["allowed"], "blocked": _stats["blocked"],
            "errors": _stats["errors"], "bytes": _stats["bytes"],
            "by_host": dict(_stats["by_host"]),
            "recent_blocks": _stats["blocks"][-20:]}
