"""Paths, constants and host-independent configuration.

Everything writable lives under one root so the app can be installed read-only
under /opt and still work.  Order of preference for that root:

    $PSE_JLAB_HOME  >  $XDG_DATA_HOME/pse-jlab  >  ~/.local/share/pse-jlab
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# --- PSE kernel constants (PSE_MASTER_ARCHITECTURE, sec. 2) -------------------
SELF_WEIGHT = 0.70
OTHER_WEIGHT_MAX = 0.30
THETA_OPTIMAL = 30.00
THETA_MIN = 25.00
THETA_MAX = 35.00
SIGMA_THETA_MIN = 0.015          # below this over a long window => schizoid screen
SIGMA_THETA_MAX = 0.030
SIGMA_FLATLINE = 0.010

APP_NAME = "pse-jlab"
APP_TITLE = "PSE J-Lab"


def _first_existing(paths) -> Path | None:
    for p in paths:
        if p and Path(p).is_dir():
            return Path(p)
    return None


def home() -> Path:
    """Writable root for state, runs, vault and logs."""
    env = os.environ.get("PSE_JLAB_HOME")
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def app_dir() -> Path:
    """Directory of the installed application package (read-only friendly)."""
    return Path(__file__).resolve().parent.parent.parent


def datasets_dir() -> Path:
    """Where MNIST / FashionMNIST live.  Searched, never assumed."""
    cand = [
        os.environ.get("PSE_JLAB_DATASETS"),
        home() / "datasets",
        app_dir().parent / "datasets",          # source checkout: <proj>/datasets
        Path("/opt") / APP_NAME / "datasets",
        Path("/usr/share") / APP_NAME / "datasets",
    ]
    found = _first_existing(cand)
    if found is not None:
        return found
    d = home() / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wheels_dir() -> Path | None:
    """Offline wheelhouse, if it shipped with the app."""
    return _first_existing([
        os.environ.get("PSE_JLAB_WHEELS"),
        app_dir().parent / "wheels",
        Path("/opt") / APP_NAME / "wheels",
    ])


def state_dir() -> Path:
    d = home() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vault_dir() -> Path:
    d = home() / "vault"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- small json settings file -------------------------------------------------

_SETTINGS_DEFAULTS = {
    "lang": "sr",
    "budget_daily_usd": 2.0,
    "providers": {},          # name -> {"model": ..., "enabled": bool}
    "roles": {},              # role -> provider name
    "polite_email": "",
    "first_run_done": False,
    "torch_optional": True,
}


def settings_path() -> Path:
    return state_dir() / "settings.json"


def load_settings() -> dict:
    p = settings_path()
    if not p.exists():
        return dict(_SETTINGS_DEFAULTS)
    try:
        data = json.loads(p.read_text())
    except Exception:
        return dict(_SETTINGS_DEFAULTS)
    merged = dict(_SETTINGS_DEFAULTS)
    merged.update(data if isinstance(data, dict) else {})
    return merged


def save_settings(data: dict) -> None:
    p = settings_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
