#!/bin/sh
# Launch PSE J-Lab, picking the first interpreter that actually has PySide6.
set -eu
here=$(cd "$(dirname "$0")" && pwd)

pick() {
    for c in "$here/.venv/bin/python" "${PYTHON:-}" python3; do
        [ -n "${c:-}" ] || continue
        [ -x "$c" ] || command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import numpy' >/dev/null 2>&1; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

py=$(pick) || {
    echo "no interpreter with numpy found. run ./setup.sh first." >&2
    exit 1
}

# datasets ship next to the app unless the user says otherwise
if [ -z "${PSE_JLAB_DATASETS:-}" ] && [ -d "$here/../datasets" ]; then
    PSE_JLAB_DATASETS="$here/../datasets"
    export PSE_JLAB_DATASETS
fi

# Display scaling override, for this app only.
#   PSE_JLAB_SCALE=1     ignore the desktop's scaling (everything gets smaller)
#   PSE_JLAB_SCALE=1.5   halfway between 100% and 200%
# The windows already size themselves to the available screen, so this is only
# needed if you want more content on screen than the desktop scaling allows.
if [ -n "${PSE_JLAB_SCALE:-}" ]; then
    QT_SCALE_FACTOR="$PSE_JLAB_SCALE"
    QT_AUTO_SCREEN_SCALE_FACTOR=0
    export QT_SCALE_FACTOR QT_AUTO_SCREEN_SCALE_FACTOR
fi

cd "$here"
exec "$py" main.py "$@"
