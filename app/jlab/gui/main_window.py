"""Main window: eight tabs and a status bar that never lies about theta."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from ..core import config, journal, theta
from .i18n import t
from .tabs import (ChatTab, GuardianTab, HomeTab, LabTab, RefsTab, ReportTab,
                   RunsTab, VaultTab)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("app_title"))
        # At 200% scaling a 1920x1080 panel is 960x540 logical pixels, so a
        # hardcoded 1180x780 opens taller and wider than the desktop.
        from .wizard import fit_to_screen
        fit_to_screen(self, 1180, 780)

        self.tabs = QTabWidget()
        # eight tabs do not fit across a 960 px logical desktop; let the tab bar
        # scroll and elide rather than pushing the last tabs off the window
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.home = HomeTab()
        self.runs = RunsTab()
        self.lab = LabTab(on_finished=self._run_finished)
        self.vault = VaultTab()
        self.refs = RefsTab()
        self.guardian = GuardianTab()
        self.chat = ChatTab()
        self.report = ReportTab()
        for w, key in ((self.home, "tab_home"), (self.lab, "tab_lab"),
                       (self.runs, "tab_runs"), (self.vault, "tab_vault"),
                       (self.refs, "tab_refs"), (self.guardian, "tab_guardian"),
                       (self.chat, "tab_chat"), (self.report, "tab_report")):
            self.tabs.addTab(w, t(key))
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self._tick()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(15000)

    def _run_finished(self, rec):
        self.runs.refresh()
        self.home.refresh()
        self._tick()

    @Slot()
    def _tick(self):
        mon = theta.get()
        mon.tick_idle()
        st = mon.status()
        colour = "#2a7" if st["healthy"] else "#c33"
        band = st["band"] if not st["flatline"] else "flatline"
        from ..llm.budget import get_budget
        b = get_budget()
        from ..guardian import egress
        eg = egress.egress_stats()
        self.status_label.setText(
            f"<span style='color:{colour}'>θ {st['theta']:.2f} ({band}, σ={st['sigma']:.3f})"
            f"</span> &nbsp;·&nbsp; {t('spent')} ${b.spent_today():.4f}/{b.daily_usd:.2f}"
            f" &nbsp;·&nbsp; egress {eg['allowed']}✓ {eg['blocked']}✗"
            f" &nbsp;·&nbsp; dnevnik {journal.get().count()}")
