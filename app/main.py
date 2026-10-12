#!/usr/bin/env python3
"""PSE J-Lab entry point.

    ./run.sh                 launch the GUI
    ./run.sh --selftest      run every check, no GUI, no network
    ./run.sh --lab <name>    run one experiment headless
    ./run.sh --list          list experiments
"""

from __future__ import annotations

import argparse
import sys


def _cli_list() -> int:
    from jlab.lab import list_experiments
    for e in list_experiments():
        print(f"{e['lab']}  {e['name']:32s}  {e['title_en']}")
    return 0


def _cli_lab(name: str, params_json: str) -> int:
    import json
    from jlab.lab import run
    params = json.loads(params_json) if params_json else {}

    def progress(frac, msg=""):
        print(f"  [{frac*100:5.1f}%] {msg}", flush=True)

    rec = run(name, params, note="cli", progress=progress)
    print(f"\n{'OK' if rec.ok else 'FAILED'}  {rec.id}")
    print(f"  {rec.seconds:.1f}s   sha256 {rec.sha256}")
    print(f"  {rec.path}")
    if not rec.ok:
        print(f"  {rec.error}")
        return 1
    if rec.result.get("claim"):
        print(f"  claim: {rec.result['claim']}")
    return 0


def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):   # θ and other non-ASCII in CLI output on Windows
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="pse-jlab", add_help=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--lab", metavar="NAME")
    ap.add_argument("--params", default="", metavar="JSON")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--diag-seal", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if getattr(args, "diag_seal", False):
        import os, tempfile, pathlib
        os.environ["PSE_JLAB_HOME"] = tempfile.mkdtemp()
        from jlab.lab import run
        from jlab.lab.runner import verify_run
        r = run("l5_logdet_consistency", {"dims": [4], "n_points": 2})
        d = pathlib.Path(r.path)
        print("run ok", r.ok)
        print("verify", verify_run(r.id))
        print("rec.sha", r.sha256)
        print("file bytes", len((d / "result.json").read_bytes()))
        print("sha sidecar", repr((d / "sha256").read_text()))
        return 0

    if args.version:
        import jlab
        print(f"PSE J-Lab {jlab.__version__}")
        return 0
    if args.selftest:
        from jlab.selftest import run_selftest
        return 0 if run_selftest() else 1
    if args.list:
        return _cli_list()
    if args.lab:
        return _cli_lab(args.lab, args.params)

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Run ./setup.sh first "
              "(it installs from the bundled wheels, no network needed).",
              file=sys.stderr)
        return 2

    from jlab.core import config
    from jlab.gui.main_window import MainWindow
    from jlab.gui.wizard import FirstRunWizard

    app = QApplication(sys.argv)
    app.setApplicationName("PSE J-Lab")
    if not config.load_settings().get("first_run_done"):
        FirstRunWizard().exec()
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
