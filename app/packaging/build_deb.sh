#!/bin/sh
# Build pse-jlab_<version>_all.deb without dpkg-deb (ar + tar only), so this
# works on a machine where dpkg-dev is not installed.
#
# The .deb carries the app, the datasets and the wheelhouse, so a fresh Mint
# machine can install and run it with no network at all.

set -eu
here=$(cd "$(dirname "$0")" && pwd)
app=$(cd "$here/.." && pwd)
proj=$(cd "$app/.." && pwd)
version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$app/jlab/__init__.py")
version=${version:-0.1.0}
pkg="pse-jlab"
build="$proj/build/${pkg}_${version}_all"
out="$proj/${pkg}_${version}_all.deb"

with_data=1
with_wheels=1
for arg in "$@"; do
    case "$arg" in
        --no-data) with_data=0 ;;
        --no-wheels) with_wheels=0 ;;
    esac
done

rm -rf "$build" "$out"
mkdir -p "$build/DEBIAN" \
         "$build/opt/$pkg/app" \
         "$build/usr/bin" \
         "$build/usr/share/applications" \
         "$build/usr/share/icons/hicolor/scalable/apps" \
         "$build/usr/share/doc/$pkg"

echo "staging application"
(cd "$app" && tar cf - \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    jlab main.py run.sh setup.sh requirements.txt packaging) \
    | (cd "$build/opt/$pkg/app" && tar xf -)

if [ "$with_data" = 1 ] && [ -d "$proj/datasets" ]; then
    echo "staging datasets"
    mkdir -p "$build/opt/$pkg/datasets"
    (cd "$proj/datasets" && tar cf - .) | (cd "$build/opt/$pkg/datasets" && tar xf -)
fi
if [ "$with_wheels" = 1 ] && [ -d "$proj/wheels" ]; then
    echo "staging wheelhouse"
    mkdir -p "$build/opt/$pkg/wheels"
    cp "$proj/wheels"/*.whl "$build/opt/$pkg/wheels/" 2>/dev/null || true
fi
for f in VODIC-SR.md GUIDE-EN.md; do
    [ -f "$proj/$f" ] && cp "$proj/$f" "$build/usr/share/doc/$pkg/"
done

cat > "$build/usr/bin/$pkg" <<'LAUNCH'
#!/bin/sh
exec /opt/pse-jlab/app/run.sh "$@"
LAUNCH
chmod 755 "$build/usr/bin/$pkg"
chmod 755 "$build/opt/$pkg/app/run.sh" "$build/opt/$pkg/app/setup.sh" \
          "$build/opt/$pkg/app/packaging/build_deb.sh" 2>/dev/null || true

cat > "$build/usr/share/applications/$pkg.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=PSE J-Lab
Comment=Jacobian geometry of neural networks
Exec=$pkg
Icon=$pkg
Terminal=false
Categories=Education;Science;Math;
DESK

cat > "$build/usr/share/icons/hicolor/scalable/apps/$pkg.svg" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="10" fill="#16202b"/>
  <text x="32" y="41" font-family="Georgia,serif" font-size="30"
        fill="#e8c37a" text-anchor="middle">J</text>
  <circle cx="15" cy="15" r="3" fill="#6fb3d2"/>
  <circle cx="49" cy="15" r="3" fill="#6fb3d2"/>
  <circle cx="15" cy="49" r="3" fill="#6fb3d2"/>
  <circle cx="49" cy="49" r="3" fill="#6fb3d2"/>
</svg>
SVG

installed_kb=$(du -sk "$build" | cut -f1)
cat > "$build/DEBIAN/control" <<CTRL
Package: $pkg
Version: $version
Section: science
Priority: optional
Architecture: all
Depends: python3 (>= 3.10), python3-venv
Recommends: python3-pip
Installed-Size: $installed_kb
Maintainer: PSE J-Lab
Description: Jacobian geometry of small neural networks
 Five laboratories built on one exact, pure-numpy Jacobian engine: dynamical
 isometry and random matrix theory, the neural tangent kernel, adversarial
 robustness, intrinsic dimension, and log|det J| in coupling flows.
 .
 Every claim is either computed (with a sealed run directory and its sha256)
 or cited (with a reference resolved over the network). Ships with MNIST and
 FashionMNIST and an offline wheelhouse, so it installs and runs with no
 network access.
CTRL

cat > "$build/DEBIAN/postinst" <<'POST'
#!/bin/sh
set -e
if [ "$1" = "configure" ]; then
    chmod 755 /opt/pse-jlab/app/run.sh /opt/pse-jlab/app/setup.sh 2>/dev/null || true
    echo "PSE J-Lab installed."
    echo "  build the environment:  /opt/pse-jlab/app/setup.sh"
    echo "  check everything:       pse-jlab --selftest"
    echo "  first launch AT HOME (the Guardian learns TLS pins there)."
fi
exit 0
POST
chmod 755 "$build/DEBIAN/postinst"

# The archives are assembled OUTSIDE the staging tree. Writing them into the
# directory being tarred makes tar fail with "file changed as we read it",
# because the archive grows inside its own input.
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
echo "2.0" > "$stage/debian-binary"
tar czf "$stage/control.tar.gz" -C "$build/DEBIAN" .
tar czf "$stage/data.tar.gz" -C "$build" --exclude='./DEBIAN' .
( cd "$stage" && ar rcs "$out" debian-binary control.tar.gz data.tar.gz )

echo
echo "built: $out"
ls -lh "$out" | awk '{print "  size:", $5}'
echo "  install: sudo apt install \"$out\""
