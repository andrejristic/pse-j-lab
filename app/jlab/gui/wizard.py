"""First run: say the one thing that must be said, then learn the TLS pins."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ..core import config
from ..guardian import policy, secrets, tls
from .i18n import t


def fit_to_screen(widget, want_w: int, want_h: int, margin: float = 0.92) -> None:
    """Never open larger than the screen actually available.

    At 200% display scaling a 1920x1080 panel gives only 960x540 logical
    pixels, so a window sized in hardcoded logical pixels can be taller than
    the desktop -- and its bottom row of buttons becomes unreachable. Ask the
    screen how much room there is instead of assuming.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        widget.resize(want_w, want_h)
        return
    avail = screen.availableGeometry()
    w = min(want_w, int(avail.width() * margin))
    h = min(want_h, int(avail.height() * margin))
    widget.resize(max(w, 320), max(h, 240))
    # let the user shrink it further; nothing here needs a large minimum
    widget.setMinimumSize(320, 240)


class FirstRunWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("wizard_title"))
        fit_to_screen(self, 680, 620)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        lay = QVBoxLayout(body)

        warn = QLabel(t("wizard_intro"))
        warn.setWordWrap(True)
        warn.setStyleSheet("background:#fff6d5; color:#663; padding:10px; "
                           "border:1px solid #dd9;")
        lay.addWidget(warn)

        lay.addWidget(QLabel(f"<b>{t('wizard_keys')}</b>"))
        form = QFormLayout()
        self.fields = {}
        for p in secrets.PROVIDERS:
            e = QLineEdit()
            e.setEchoMode(QLineEdit.Password)
            e.setText(secrets.get_key(p))
            e.setPlaceholderText("(prazno = ovaj provajder se ne koristi)")
            form.addRow(p, e)
            self.fields[p] = e
        lay.addLayout(form)

        self.lang = QCheckBox("English interface (unchecked = srpski)")
        self.lang.setChecked(config.load_settings().get("lang", "sr") == "en")
        lay.addWidget(self.lang)

        lay.addWidget(QLabel(f"<b>{t('wizard_learn')}</b> — "
                             f"{len(policy.ALLOWLIST)} dozvoljenih domena"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        lay.addWidget(self.log, 1)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)

        # The buttons live OUTSIDE the scroll area, pinned to the bottom of the
        # dialog. Inside it they scroll away, and on a short screen the user
        # cannot reach "Finish" at all.
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 10, 10)
        self.btn_learn = QPushButton(t("wizard_learn"))
        self.btn_learn.clicked.connect(self._learn)
        self.btn_done = QPushButton(t("wizard_finish"))
        self.btn_done.setDefault(True)
        self.btn_done.clicked.connect(self._finish)
        row.addWidget(self.btn_learn)
        row.addStretch(1)
        row.addWidget(self.btn_done)
        outer.addLayout(row)

    @Slot()
    def _learn(self):
        self.btn_learn.setEnabled(False)
        hosts = sorted(policy.ALLOWLIST)
        self.bar.setRange(0, len(hosts))
        for i, h in enumerate(hosts, 1):
            try:
                v = tls.check_and_pin(h)
                self.log.appendPlainText(f"{v['verdict']:9s} {h}")
            except Exception as e:
                self.log.appendPlainText(f"{'greska':9s} {h}: {type(e).__name__}: {e}")
            self.bar.setValue(i)
        self.btn_learn.setEnabled(True)

    @Slot()
    def _finish(self):
        for p, e in self.fields.items():
            secrets.set_key(p, e.text().strip())
        s = config.load_settings()
        s["lang"] = "en" if self.lang.isChecked() else "sr"
        s["first_run_done"] = True
        config.save_settings(s)
        self.accept()
