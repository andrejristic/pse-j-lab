#!/bin/sh
# Build the virtual environment for PSE J-Lab.
#
# Mint blocks pip from touching the system Python (PEP 668), so everything goes
# into app/.venv. If the bundled wheelhouse is present the install runs fully
# offline -- which is the point: nothing here should need a hotel's wifi.

set -eu
here=$(cd "$(dirname "$0")" && pwd)
venv="$here/.venv"
wheels="$here/../wheels"

py=${PYTHON:-python3}
if ! "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
    echo "need Python 3.10 or newer; found: $("$py" -V 2>&1)" >&2
    exit 1
fi

if [ ! -d "$venv" ]; then
    echo "creating $venv"
    "$py" -m venv "$venv"
fi
. "$venv/bin/activate"

pkgs="numpy matplotlib PySide6-Essentials"
extra="${1:-}"
if [ "$extra" = "--with-torch" ]; then
    pkgs="$pkgs torch"
fi

if [ -d "$wheels" ] && [ "$(ls -A "$wheels" 2>/dev/null)" ]; then
    echo "installing offline from $wheels"
    python -m pip install --quiet --upgrade --no-index --find-links "$wheels" pip || true
    python -m pip install --no-index --find-links "$wheels" $pkgs
else
    echo "no bundled wheels found -- falling back to the network"
    python -m pip install --quiet --upgrade pip || true
    python -m pip install $pkgs
fi

echo
python - <<'PY'
import importlib
for m in ("numpy", "matplotlib", "PySide6"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({e})")
try:
    import torch
    print(f"  {'torch':12s} {torch.__version__} (optional oracle)")
except Exception:
    print(f"  {'torch':12s} not installed (optional; ./setup.sh --with-torch)")
PY
echo
echo "done. now: ./run.sh --selftest"
