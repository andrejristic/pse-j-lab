"""Reproducible experiment runs.

Every run produces a directory containing everything needed to re-run it and
nothing that depends on when it was run:

    params.json     the exact resolved parameters (defaults merged in)
    result.json     the numbers
    meta.json       library versions, timings, host fingerprint
    sha256          hash of result.json -- what a COMPUTED card must cite
    plots/*.png     optional figures

A claim that cannot point at one of these directories does not get to be a
COMPUTED card.  That is the whole contract between this module and the vault.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core import config, journal, theta
from .registry import get_experiment


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def param_fingerprint(name: str, params: dict) -> str:
    return hashlib.sha256(_canon({"name": name, "params": params}).encode()).hexdigest()[:12]


@dataclass
class RunRecord:
    id: str
    name: str
    lab: str
    params: dict
    result: dict
    sha256: str
    path: str
    seconds: float
    started_utc: str
    ok: bool = True
    error: str = ""
    plots: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _env_meta() -> dict:
    meta = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch  # noqa: F401
        meta["torch"] = torch.__version__
    except Exception:
        meta["torch"] = None
    return meta


def run(name: str, params: dict | None = None, note: str = "",
        progress=None, make_plots: bool = True) -> RunRecord:
    """Execute an experiment and seal the result on disk."""
    exp = get_experiment(name)
    resolved = dict(exp["defaults"])
    resolved.update(params or {})
    fp = param_fingerprint(name, resolved)
    started = time.gmtime()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", started)
    rundir = config.runs_dir() / f"{name}-{stamp}-{fp}"
    rundir.mkdir(parents=True, exist_ok=True)

    (rundir / "params.json").write_text(json.dumps(resolved, indent=2, default=_default))
    t0 = time.perf_counter()
    ok, err, result = True, "", {}
    try:
        result = exp["fn"](progress=progress, **resolved)
    except Exception as e:                      # a failed run is still a record
        ok, err = False, f"{type(e).__name__}: {e}"
        result = {"error": err}
    secs = time.perf_counter() - t0

    payload = json.dumps(result, indent=2, default=_default, sort_keys=True)
    (rundir / "result.json").write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (rundir / "sha256").write_text(digest + "  result.json\n")
    (rundir / "meta.json").write_text(json.dumps({
        "id": rundir.name, "name": name, "lab": exp["lab"], "note": note,
        "seconds": secs, "started_utc": stamp, "ok": ok, "error": err,
        "env": _env_meta(), "fingerprint": fp,
    }, indent=2))

    plots = []
    if make_plots and ok and exp.get("plot"):
        try:
            from . import plots as plotmod
            plots = plotmod.render(exp["plot"], result, rundir / "plots")
        except Exception as e:                  # plotting must never kill a run
            (rundir / "plots_error.txt").write_text(f"{type(e).__name__}: {e}")

    rec = RunRecord(id=rundir.name, name=name, lab=exp["lab"], params=resolved,
                    result=result, sha256=digest, path=str(rundir), seconds=secs,
                    started_utc=stamp, ok=ok, error=err, plots=plots)
    journal.log("LAB_RUN", {"id": rec.id, "name": name, "lab": exp["lab"],
                            "sha256": digest, "seconds": round(secs, 3),
                            "ok": ok, "error": err, "note": note}, actor="lab")
    # an experiment that could have refuted us is engagement with the other
    theta.bump("lab_run", metadata={"experiment": name, "ok": ok})
    return rec


def load_run(run_id: str) -> RunRecord | None:
    d = config.runs_dir() / run_id
    if not d.is_dir():
        return None
    try:
        meta = json.loads((d / "meta.json").read_text())
        result = json.loads((d / "result.json").read_text())
        params = json.loads((d / "params.json").read_text())
        digest = (d / "sha256").read_text().split()[0]
    except Exception:
        return None
    plots = sorted(str(p) for p in (d / "plots").glob("*.png")) if (d / "plots").is_dir() else []
    return RunRecord(id=d.name, name=meta.get("name", ""), lab=meta.get("lab", ""),
                     params=params, result=result, sha256=digest, path=str(d),
                     seconds=meta.get("seconds", 0.0),
                     started_utc=meta.get("started_utc", ""),
                     ok=meta.get("ok", True), error=meta.get("error", ""), plots=plots)


def verify_run(run_id: str) -> dict:
    """Re-hash result.json and compare with the sealed digest."""
    d = config.runs_dir() / run_id
    if not d.is_dir():
        return {"ok": False, "reason": "no such run"}
    payload = (d / "result.json").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sealed = (d / "sha256").read_text().split()[0]
    return {"ok": digest == sealed, "computed": digest, "sealed": sealed}


def list_runs(name: str | None = None, limit: int = 200) -> list:
    out = []
    for d in sorted(config.runs_dir().glob("*"), reverse=True):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            continue
        if name and meta.get("name") != name:
            continue
        out.append({"id": d.name, "name": meta.get("name"), "lab": meta.get("lab"),
                    "seconds": meta.get("seconds"), "ok": meta.get("ok", True),
                    "started_utc": meta.get("started_utc", ""),
                    "note": meta.get("note", "")})
        if len(out) >= limit:
            break
    return out
