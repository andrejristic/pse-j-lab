"""Small widgets: LaTeX-to-PNG via mathtext, a JSON viewer, an image pane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from PySide6.QtCore import Qt                          # noqa: E402
from PySide6.QtGui import QPixmap                      # noqa: E402
from PySide6.QtWidgets import (QLabel, QScrollArea, QTreeWidget,  # noqa: E402
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import config                              # noqa: E402


def render_math(expr: str, fontsize: int = 15, dpi: int = 110) -> Path:
    """Render a mathtext expression to a cached PNG.  No QtWebEngine, no LaTeX."""
    cache = config.cache_dir() / "math"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{expr}|{fontsize}|{dpi}".encode()).hexdigest()[:20]
    out = cache / f"{key}.png"
    if out.exists():
        return out
    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    fig.text(0, 0, expr, fontsize=fontsize)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, transparent=False)
    plt.close(fig)
    return out


class MathLabel(QLabel):
    """A label whose text is a mathtext formula rendered to a pixmap."""

    def __init__(self, expr: str = "", fontsize: int = 15, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        if expr:
            self.set_expr(expr, fontsize)

    def set_expr(self, expr: str, fontsize: int = 15) -> None:
        try:
            self.setPixmap(QPixmap(str(render_math(expr, fontsize))))
        except Exception:
            self.setText(expr)


class ImagePane(QScrollArea):
    """Scrollable stack of a run's plots, scaled to the width of the pane.

    A figure rendered at a fixed pixel size is small on a HiDPI screen, so the
    default is to fit the available width (never enlarging past the original,
    which would only blur it). Toggle `fit` off for 1:1 pixels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._inner = QWidget()
        self._lay = QVBoxLayout(self._inner)
        self._lay.setAlignment(Qt.AlignTop)
        self.setWidget(self._inner)
        self._pixmaps = []
        self._labels = []
        self.fit = True

    def clear(self) -> None:
        self._pixmaps = []
        self._labels = []
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def set_fit(self, on: bool) -> None:
        self.fit = bool(on)
        self._rescale()

    def show_paths(self, paths) -> None:
        self.clear()
        if not paths:
            lbl = QLabel("(nema grafika za ovaj rezultat / no plots for this run)")
            lbl.setAlignment(Qt.AlignCenter)
            self._lay.addWidget(lbl)
            return
        for p in paths:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            pm = QPixmap(str(p))
            if pm.isNull():
                lbl.setText(str(p))
            else:
                self._pixmaps.append(pm)
                self._labels.append(lbl)
            self._lay.addWidget(lbl)
        self._rescale()

    def _rescale(self) -> None:
        avail = max(120, self.viewport().width() - 24)
        for pm, lbl in zip(self._pixmaps, self._labels):
            if self.fit and pm.width() > avail:
                lbl.setPixmap(pm.scaledToWidth(avail, Qt.SmoothTransformation))
            else:
                lbl.setPixmap(pm)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class JsonTree(QTreeWidget):
    """Read-only tree view of a nested dict; long lists are truncated."""

    MAX_ITEMS = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["", ""])
        self.setAlternatingRowColors(True)

    def load(self, data) -> None:
        self.clear()
        self._add(self.invisibleRootItem(), data)
        self.expandToDepth(1)
        self.resizeColumnToContents(0)

    def _add(self, parent, data, key: str = "") -> None:
        if isinstance(data, dict):
            node = QTreeWidgetItem([key or "{}", f"{len(data)} polja"])
            parent.addChild(node)
            for k, v in data.items():
                self._add(node, v, str(k))
        elif isinstance(data, (list, tuple)):
            n = len(data)
            node = QTreeWidgetItem([key or "[]", f"{n} stavki"])
            parent.addChild(node)
            for i, v in enumerate(data[:self.MAX_ITEMS]):
                self._add(node, v, f"[{i}]")
            if n > self.MAX_ITEMS:
                node.addChild(QTreeWidgetItem(["...", f"jos {n - self.MAX_ITEMS}"]))
        else:
            if isinstance(data, float):
                text = f"{data:.6g}"
            else:
                text = str(data)
            if len(text) > 160:
                text = text[:160] + "..."
            parent.addChild(QTreeWidgetItem([key, text]))


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
