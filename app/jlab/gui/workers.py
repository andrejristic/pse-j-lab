"""Background jobs.

The signals are declared on the QThread subclass and connected to slots that
live on widgets in the main thread, so Qt delivers them queued. An earlier
generation of this code passed a closure as the callback, which produced a
DIRECT connection: the callback then ran on the worker thread and the worker
waited on itself. Hence: bound signals only, never a closure across threads.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal


def resolve_call_kwargs(fn, kwargs: dict, progress_cb) -> dict:
    """Decide how a job function is called, and never leak the marker.

    Kept separate from the thread so it can be tested without Qt: this is the
    line that broke every progress-reporting button in the app when it was an
    `or` expression, because Python skipped the `pop` on short-circuit.
    """
    out = dict(kwargs)
    wants = bool(out.pop("_wants_progress", False))
    code = getattr(fn, "__code__", None)
    takes_progress = bool(code and "progress" in code.co_varnames)
    if wants or takes_progress:
        out["progress"] = progress_cb
    return out


class Job(QThread):
    progress = Signal(float, str)      # fraction 0..1, message
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent: QObject | None = None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, frac: float, msg: str = "") -> None:
        # called from inside the worker function; emitting is thread-safe
        self.progress.emit(float(frac), str(msg))

    def run(self) -> None:
        try:
            kwargs = resolve_call_kwargs(self._fn, self._kwargs, self._progress)
            result = self._fn(*self._args, **kwargs)
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=6)}")


class LabJob(Job):
    """Runs a lab experiment; the runner accepts a `progress` callable."""

    def __init__(self, name: str, params: dict, note: str = "", parent=None):
        from ..lab import run
        super().__init__(run, name, params, note=note, parent=parent)
        self._kwargs["_wants_progress"] = True
