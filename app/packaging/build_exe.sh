#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"; REPO="$(dirname "$APP")"
PY="${PYTHON:-$APP/.venv/Scripts/python.exe}"
VERSION="${1:-0.1.0}"; PKG="pse-jlab"
STAGE="$HERE/stage-win"; OUT="$REPO/dist"
ISCC="${ISCC:-/c/Program Files (x86)/Inno Setup 6/ISCC.exe}"
echo "== PyInstaller bundle"
rm -rf "$APP/build/pyi" "$STAGE"
"$PY" -m PyInstaller --noconfirm --clean --onedir --windowed --name "$PKG" \
  --icon "$HERE/$PKG.ico" \
  --distpath "$APP/build/pyi/dist" --workpath "$APP/build/pyi/work" --specpath "$APP/build/pyi" \
  --collect-submodules jlab --collect-data jlab \
  --collect-all matplotlib \
  --add-data "$REPO/datasets;datasets" \
  "$APP/main.py"
APPDIR="$STAGE/$PKG"; mkdir -p "$APPDIR"
cp -a "$APP/build/pyi/dist/$PKG/." "$APPDIR/"
cp "$HERE/$PKG.ico" "$APPDIR/"; cp "$REPO/LICENSE" "$APPDIR/LICENSE.txt"; cp "$REPO/NOTICE" "$APPDIR/NOTICE.txt"
echo "== Inno Setup"; mkdir -p "$OUT"
MSYS2_ARG_CONV_EXCL="*" "$ISCC" "/DAppVersion=$VERSION" "/DSourceDir=$(cygpath -w "$APPDIR")" "/DOutDir=$(cygpath -w "$OUT")" \
  "$(cygpath -w "$HERE/$PKG.iss")"
echo "built $OUT/PSEJLab-Setup-$VERSION.exe"
