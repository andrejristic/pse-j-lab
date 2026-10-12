"""The eight tabs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextEdit,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import config, journal, theta
from ..data.idx import dataset_status
from ..lab import list_experiments, list_runs, run as run_experiment
from ..lab.registry import EXPERIMENTS, LABS
from ..lab.runner import load_run, verify_run
from ..vault import CardKind, GateError, admit, get_store, refute, wound
from ..vault.seed import load_proposals, load_topics
from .i18n import t
from .widgets import ImagePane, JsonTree, MathLabel, pretty_json
from .workers import Job, LabJob


# --------------------------------------------------------------------------- #
class HomeTab(QWidget):
    """What this app is, what is on this machine, and what to do first."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        head = QLabel("<h2>PSE J-Lab</h2>"
                      "<p>Jakobijani u masinskom ucenju — pet laboratorija, "
                      "jedan egzaktan engine, jedna kapija u trezor.</p>")
        head.setWordWrap(True)
        lay.addWidget(head)
        lay.addWidget(MathLabel(r"$J(x) = W_L\,D_{L-1}(x)\,W_{L-1}\cdots D_1(x)\,W_1$", 16))

        self.labs = QTreeWidget()
        self.labs.setColumnCount(2)
        self.labs.setHeaderLabels(["Lab", "sta meri"])
        for code, names in LABS.items():
            node = QTreeWidgetItem([f"{code} — {names['sr']}", ""])
            for e in list_experiments(code):
                node.addChild(QTreeWidgetItem([e["title_sr"], e["doc"]]))
            self.labs.addTopLevelItem(node)
        self.labs.expandAll()
        self.labs.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        lay.addWidget(self.labs, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.refresh()

    def refresh(self) -> None:
        ds = dataset_status()
        man = ds["manifest"]
        ok = man.get("ok")
        parts = [f"<b>Podaci:</b> {ds['root']} — "
                 f"{'<span style=\"color:#2a7\">' + t('datasets_ok') + '</span>' if ok else '<span style=\"color:#c33\">' + t('datasets_bad') + '</span>'}"]
        for name, d in ds["datasets"].items():
            parts.append(f"{name}: {'kompletan' if d['complete'] else 'NEDOSTAJE'}")
        parts.append(f"<b>Prostor za rad:</b> {config.home()}")
        parts.append(f"<b>Pokretanja do sada:</b> {len(list_runs(limit=9999))}")
        self.status.setText(" &nbsp;·&nbsp; ".join(parts))


# --------------------------------------------------------------------------- #
class LabTab(QWidget):
    """Pick an experiment, edit its parameters, run it."""

    def __init__(self, on_finished=None, parent=None):
        super().__init__(parent)
        self._job: Job | None = None
        self._on_finished = on_finished
        split = QSplitter(Qt.Horizontal, self)

        left = QWidget()
        llay = QVBoxLayout(left)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([t("experiment")])
        for code, names in LABS.items():
            node = QTreeWidgetItem([f"{code} — {names['sr']}"])
            for e in list_experiments(code):
                child = QTreeWidgetItem([e["title_sr"]])
                child.setData(0, Qt.UserRole, e["name"])
                child.setToolTip(0, e["doc"])
                node.addChild(child)
            self.tree.addTopLevelItem(node)
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._on_pick)
        llay.addWidget(self.tree)

        right = QWidget()
        rlay = QVBoxLayout(right)
        self.title = QLabel("<i>izaberi eksperiment levo</i>")
        self.title.setWordWrap(True)
        rlay.addWidget(self.title)
        rlay.addWidget(QLabel(t("params") + " (JSON):"))
        self.params = QPlainTextEdit()
        self.params.setPlaceholderText("{}")
        rlay.addWidget(self.params, 1)

        btns = QHBoxLayout()
        self.btn_run = QPushButton(t("run"))
        self.btn_run.clicked.connect(self._start)
        self.btn_defaults = QPushButton(t("defaults"))
        self.btn_defaults.clicked.connect(self._reset_params)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_defaults)
        btns.addStretch(1)
        rlay.addLayout(btns)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        rlay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        rlay.addWidget(self.log, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([320, 640])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

    def _current_name(self) -> str | None:
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it is not None else None

    @Slot()
    def _on_pick(self, *_):
        name = self._current_name()
        if not name:
            return
        e = EXPERIMENTS[name]
        self.title.setText(f"<b>{e['title_sr']}</b><br>{e['doc']}<br>"
                           f"<i>{e['claim'] if 'claim' in e else ''}</i>")
        self._reset_params()

    @Slot()
    def _reset_params(self):
        name = self._current_name()
        if name:
            self.params.setPlainText(pretty_json(EXPERIMENTS[name]["defaults"]))

    @Slot()
    def _start(self):
        name = self._current_name()
        if not name:
            return
        try:
            params = json.loads(self.params.toPlainText() or "{}")
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "JSON", f"Parametri nisu ispravan JSON:\n{e}")
            return
        self.btn_run.setEnabled(False)
        self.bar.setValue(0)
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {name} {t('running')}")
        self._job = LabJob(name, params, note="gui")
        self._job.progress.connect(self._on_progress)
        self._job.finished_ok.connect(self._on_ok)
        self._job.failed.connect(self._on_fail)
        self._job.start()

    @Slot(float, str)
    def _on_progress(self, frac: float, msg: str):
        self.bar.setValue(int(frac * 100))
        if msg:
            self.log.appendPlainText(f"   {msg}")

    @Slot(object)
    def _on_ok(self, rec):
        self.bar.setValue(100)
        self.btn_run.setEnabled(True)
        status = t("done") if rec.ok else t("failed")
        self.log.appendPlainText(
            f"[{time.strftime('%H:%M:%S')}] {status} — {rec.id}  "
            f"({rec.seconds:.1f}s, sha {rec.sha256[:12]})")
        if not rec.ok:
            self.log.appendPlainText(f"   {rec.error}")
        if self._on_finished:
            self._on_finished(rec)

    @Slot(str)
    def _on_fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.log.appendPlainText(msg)


# --------------------------------------------------------------------------- #
class RunsTab(QWidget):
    """Every sealed run: its numbers, its plots, its hash, and the card path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        split = QSplitter(Qt.Horizontal, self)
        left = QWidget()
        llay = QVBoxLayout(left)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_pick)
        llay.addWidget(self.list, 1)
        btns = QHBoxLayout()
        b_refresh = QPushButton("Osvezi")
        b_refresh.clicked.connect(self.refresh)
        b_verify = QPushButton(t("verify"))
        b_verify.clicked.connect(self._verify)
        b_card = QPushButton(t("make_card"))
        b_card.clicked.connect(self._make_card)
        for b in (b_refresh, b_verify, b_card):
            btns.addWidget(b)
        llay.addLayout(btns)
        btns2 = QHBoxLayout()
        b_plot = QPushButton("Ponovo iscrtaj")
        b_plot.setToolTip("Redraws the figures from the sealed result.json at the "
                          "current resolution. The numbers and the sha256 are "
                          "untouched.")
        b_plot.clicked.connect(self._rerender)
        self.chk_fit = QCheckBox("prilagodi sirini")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(lambda on: self.images.set_fit(on))
        btns2.addWidget(b_plot)
        btns2.addWidget(self.chk_fit)
        llay.addLayout(btns2)

        right = QSplitter(Qt.Vertical)
        self.tree = JsonTree()
        self.images = ImagePane()
        right.addWidget(self.tree)
        right.addWidget(self.images)
        right.setSizes([320, 380])

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([300, 660])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        for r in list_runs(limit=400):
            mark = "" if r.get("ok", True) else "✗ "
            item = QListWidgetItem(f"{mark}{r['lab']} {r['name']}\n   "
                                   f"{r['started_utc']}  {r.get('seconds', 0):.1f}s")
            item.setData(Qt.UserRole, r["id"])
            self.list.addItem(item)

    def _current_id(self) -> str | None:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    @Slot()
    def _on_pick(self, *_):
        rid = self._current_id()
        if not rid:
            return
        rec = load_run(rid)
        if rec is None:
            return
        self.tree.load({"id": rec.id, "sha256": rec.sha256, "seconds": rec.seconds,
                        "params": rec.params, "result": rec.result})
        self.images.show_paths(rec.plots)

    @Slot()
    def _rerender(self):
        rid = self._current_id()
        if not rid:
            return
        from ..lab.plots import rerender_run
        r = rerender_run(rid)
        if not r["ok"]:
            QMessageBox.warning(self, "Ponovo iscrtaj", r["reason"])
            return
        if not r["plots"]:
            QMessageBox.information(self, "Ponovo iscrtaj",
                                    r.get("note", "nema sta da se crta"))
            return
        self.images.show_paths(r["plots"])
        QMessageBox.information(self, "Ponovo iscrtaj",
                                f"{len(r['plots'])} grafik(a) na {r['dpi']} dpi.\n"
                                f"Brojevi i sha256 nisu dirani.")

    @Slot()
    def _verify(self):
        rid = self._current_id()
        if not rid:
            return
        v = verify_run(rid)
        QMessageBox.information(
            self, t("verify"),
            ("Hes se poklapa — rezultat nije menjan.\n\n" if v["ok"]
             else "HES SE NE POKLAPA — result.json je menjan posle pecacenja!\n\n")
            + f"sealed:   {v.get('sealed','')}\ncomputed: {v.get('computed','')}")

    @Slot()
    def _make_card(self):
        rid = self._current_id()
        if not rid:
            return
        rec = load_run(rid)
        if rec is None:
            return
        claim, ok = QInputDialog.getMultiLineText(
            self, t("make_card"), t("claim") + f"\n({t('claim_hint')})",
            rec.result.get("claim", "") if isinstance(rec.result, dict) else "")
        if not ok or not claim.strip():
            return
        try:
            c = admit(CardKind.COMPUTED, claim, run_id=rec.id, topic=rec.lab,
                      body=f"{rec.name} · {rec.started_utc}", author="user")
            QMessageBox.information(self, t("make_card"),
                                    f"Upisano: {c.id}\nvezano za {rec.id}\nsha {c.sha256[:16]}")
        except GateError as e:
            QMessageBox.warning(self, "Kapija je odbila", str(e))


# --------------------------------------------------------------------------- #
class VaultTab(QWidget):
    """Cards, and the two ways to damage one."""

    KINDS = ("(sve)",) + CardKind.ALL

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(self.KINDS)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.query = QLineEdit()
        self.query.setPlaceholderText(t("search"))
        self.query.returnPressed.connect(self.refresh)
        b = QPushButton(t("search"))
        b.clicked.connect(self.refresh)
        top.addWidget(QLabel(t("kind") + ":"))
        top.addWidget(self.kind)
        top.addWidget(self.query, 1)
        top.addWidget(b)
        lay.addLayout(top)

        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([t("kind"), t("claim"), t("topic"), "id"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_pick)
        split.addWidget(self.table)

        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        blay.addWidget(self.detail, 1)
        btns = QHBoxLayout()
        b_w = QPushButton(t("wound"))
        b_w.clicked.connect(lambda: self._attack(False))
        b_r = QPushButton(t("refute"))
        b_r.clicked.connect(lambda: self._attack(True))
        btns.addWidget(b_w)
        btns.addWidget(b_r)
        btns.addStretch(1)
        blay.addLayout(btns)
        split.addWidget(bottom)
        split.setSizes([420, 260])
        lay.addWidget(split)
        self.refresh()

    def refresh(self) -> None:
        store = get_store()
        q = self.query.text().strip()
        kind = self.kind.currentText()
        cards = store.search(q) if q else store.list(
            kind=None if kind == "(sve)" else kind, limit=500)
        if q and kind != "(sve)":
            cards = [c for c in cards if c.kind == kind]
        self.table.setRowCount(len(cards))
        for i, c in enumerate(cards):
            for j, val in enumerate((c.kind, c.claim, c.topic, c.id)):
                self.table.setItem(i, j, QTableWidgetItem(val))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _selected_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 3)
        return item.text() if item else None

    @Slot()
    def _on_pick(self):
        cid = self._selected_id()
        if not cid:
            return
        c = get_store().get(cid)
        if c is None:
            return
        html = [f"<b>{c.kind}</b> — {c.claim}",
                f"<br><i>tema:</i> {c.topic or '-'} &nbsp; <i>autor:</i> {c.author}"]
        if c.run_id:
            html.append(f"<br><i>run:</i> {c.run_id} &nbsp; <i>sha256:</i> {c.sha256[:24]}")
        if c.reference:
            html.append(f"<br><i>referenca:</i> {c.reference}")
        if c.body:
            html.append(f"<br><br>{c.body}")
        if c.attacks:
            html.append("<br><br><b>napadi:</b>")
            for a in c.attacks:
                html.append(f"<br>· [{a.get('verdict','')}] {a.get('by','')}: {a.get('argument','')}")
        self.detail.setHtml("".join(html))

    def _attack(self, kill: bool):
        cid = self._selected_id()
        if not cid:
            return
        arg, ok = QInputDialog.getMultiLineText(self, t("argument"), t("argument") + ":")
        if not ok or not arg.strip():
            return
        try:
            (refute if kill else wound)(cid, "user", arg.strip())
        except GateError as e:
            QMessageBox.warning(self, "greska", str(e))
            return
        self.refresh()
        self._on_pick()


# --------------------------------------------------------------------------- #
class RefsTab(QWidget):
    """Proposed citations, and the network check that turns them into cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Job | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Citati nisu kartice dok se ne razrese preko mreze. "
            "Naslov koji se ne poklopi znaci da je ID pogresno zapamcen — "
            "takav predlog se odbija, ne ispravlja."))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["status", "arXiv", "ocekivano", "stvarno", "slicnost"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.btn = QPushButton(t("verify_all"))
        self.btn.clicked.connect(self._verify)
        self.chk = QPushButton(t("admit_verified"))
        self.chk.clicked.connect(lambda: self._verify(admit_cards=True))
        btns.addWidget(self.btn)
        btns.addWidget(self.chk)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.refresh()

    def refresh(self) -> None:
        rows = load_proposals()
        self.table.setRowCount(len(rows))
        for i, p in enumerate(rows):
            sim = p.get("title_similarity")
            vals = (p.get("status", "unverified"), p.get("arxiv", ""),
                    p.get("title_guess", ""), p.get("resolved_title", "") or p.get("error", ""),
                    "" if sim is None else f"{sim:.2f}")
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))

    def _verify(self, admit_cards: bool = False):
        from ..llm import verify_all_proposals
        self.btn.setEnabled(False)
        self.chk.setEnabled(False)
        self._job = Job(verify_all_proposals, admit_cards=admit_cards, _wants_progress=True)
        self._job.progress.connect(lambda f, m: self.bar.setValue(int(f * 100)))
        self._job.finished_ok.connect(self._done)
        self._job.failed.connect(self._fail)
        self._job.start()

    @Slot(object)
    def _done(self, r):
        self.btn.setEnabled(True)
        self.chk.setEnabled(True)
        self.refresh()
        QMessageBox.information(
            self, t("verify_all"),
            f"potvrdjeno {r['verified']}/{r['n']}\n"
            f"naslov se ne poklapa: {r['mismatched']}\n"
            f"nerazreseno: {r['unresolved']}\n"
            f"upisano kartica: {len(r['admitted'])}")

    @Slot(str)
    def _fail(self, msg: str):
        self.btn.setEnabled(True)
        self.chk.setEnabled(True)
        QMessageBox.warning(self, "greska", msg)


# --------------------------------------------------------------------------- #
class GuardianTab(QWidget):
    """Mostly a monitor — but it owns the one action that must happen at home:
    learning the TLS fingerprint of every host the app is allowed to reach.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Job | None = None
        lay = QVBoxLayout(self)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        lay.addWidget(self.summary)

        row = QHBoxLayout()
        self.btn_pin = QPushButton("Nauci TLS otiske (uradi ovo KOD KUCE)")
        self.btn_pin.clicked.connect(self._learn_pins)
        b2 = QPushButton("Osvezi")
        b2.clicked.connect(self.refresh)
        row.addWidget(self.btn_pin, 1)
        row.addWidget(b2)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)

        split = QSplitter(Qt.Vertical)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QTextEdit.NoWrap)
        split.addWidget(self.view)

        box = QGroupBox("Probaj skener na tekstu (prompt injection)")
        blay = QVBoxLayout(box)
        self.probe = QPlainTextEdit()
        self.probe.setPlaceholderText("nalepi ovde tekst iz nekog rada...")
        blay.addWidget(self.probe)
        b = QPushButton("Skeniraj")
        b.clicked.connect(self._scan)
        blay.addWidget(b)
        self.scan_out = QLabel("-")
        self.scan_out.setWordWrap(True)
        blay.addWidget(self.scan_out)
        split.addWidget(box)
        split.setSizes([420, 240])
        lay.addWidget(split, 1)
        self.refresh()

    def refresh(self) -> None:
        from ..guardian import get_guardian, policy
        from ..guardian.tls import load_pins
        st = get_guardian().status()
        pins = load_pins()
        total = len(policy.ALLOWLIST)
        missing = sorted(set(policy.ALLOWLIST) - set(pins))
        eg = st["egress"]
        parts = [f"<b>Otisci:</b> {len(pins)}/{total} domena pinovano"]
        if missing:
            parts.append(
                "<span style='color:#c33'><b>{} jos nije pinovano</b></span> — "
                "klikni dugme ispod dok si na mrezi kojoj verujes.<br>"
                "<small>{}</small>".format(len(missing), ", ".join(missing)))
        else:
            parts.append("<span style='color:#2a7'>svi domeni pinovani</span>")
        parts.append(f"<b>Saobracaj:</b> {eg['allowed']} propusteno, "
                     f"{eg['blocked']} blokirano, {eg['errors']} gresaka")
        parts.append(f"<b>Intervencije:</b> {st['interventions']}")
        note = st.get("vigilance_note")
        if note:
            parts.append(f"<span style='color:#c80'>{note}</span>")
        post = st["posture"]
        if not post.get("keys_file_ok"):
            parts.append("<span style='color:#c33'>keys.json nije 0600</span>")
        self.summary.setText("<br>".join(parts))
        self.view.setPlainText(pretty_json(st))

    @Slot()
    def _learn_pins(self):
        from ..guardian import policy
        from ..guardian.tls import check_and_pin
        hosts = sorted(policy.ALLOWLIST)

        def work(progress=None):
            out = []
            for i, h in enumerate(hosts, 1):
                try:
                    v = check_and_pin(h)
                    out.append((h, v.get("verdict", "?")))
                except Exception as e:
                    out.append((h, f"greska: {type(e).__name__}"))
                if progress:
                    progress(i / len(hosts), h)
            return out

        self.btn_pin.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self._job = Job(work, _wants_progress=True)
        self._job.progress.connect(lambda f, m: self.bar.setValue(int(f * 100)))
        self._job.finished_ok.connect(self._pins_done)
        self._job.failed.connect(self._pins_failed)
        self._job.start()

    @Slot(object)
    def _pins_done(self, out):
        self.btn_pin.setEnabled(True)
        self.bar.setVisible(False)
        self.refresh()
        good = sum(1 for _, v in out if v in ("learned", "match", "rotated"))
        detail = "\n".join(f"{v:10s} {h}" for h, v in out)
        QMessageBox.information(self, "TLS otisci",
                                f"{good}/{len(out)} u redu\n\n{detail}")

    @Slot(str)
    def _pins_failed(self, msg: str):
        self.btn_pin.setEnabled(True)
        self.bar.setVisible(False)
        QMessageBox.warning(self, "greska", msg.splitlines()[0])

    @Slot()
    def _scan(self):
        from ..guardian.content import scan_injection
        r = scan_injection(self.probe.toPlainText())
        colour = {"benign": "#2a7", "elevated": "#c80", "critical": "#c33"}[r["risk"]]
        self.scan_out.setText(
            f"<b style='color:{colour}'>{r['risk'].upper()}</b> — {r['n']} nalaza "
            f"({', '.join(r['labels']) or 'nema'})")


# --------------------------------------------------------------------------- #
class ChatTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Job | None = None
        self._history = []
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.role = QComboBox()
        from ..llm.router import ROLES
        self.role.addItems(ROLES)
        top.addWidget(QLabel(t("role") + ":"))
        top.addWidget(self.role)
        self.route_note = QLabel("")
        top.addWidget(self.route_note, 1)
        lay.addLayout(top)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        lay.addWidget(self.view, 1)
        self.entry = QPlainTextEdit()
        self.entry.setMaximumHeight(110)
        lay.addWidget(self.entry)
        self.btn = QPushButton(t("send"))
        self.btn.clicked.connect(self._send)
        lay.addWidget(self.btn)
        self._refresh_route()

    def _refresh_route(self):
        from ..llm import get_router
        r = get_router()
        avail = r.available()
        if not avail:
            self.route_note.setText(f"<i>{t('no_key')} — {t('offline')}</i>")
        else:
            sel = r.pick(self.role.currentText())
            self.route_note.setText(f"<i>{sel['provider']} — {sel['reason']}</i>")

    @Slot()
    def _send(self):
        text = self.entry.toPlainText().strip()
        if not text:
            return
        from ..llm import get_router
        self.view.append(f"<b>ti:</b> {text}")
        self.entry.clear()
        self.btn.setEnabled(False)
        role = self.role.currentText()
        self._job = Job(get_router().ask, role, text, history=list(self._history))
        self._job.finished_ok.connect(self._ok)
        self._job.failed.connect(self._fail)
        self._job.start()
        self._pending = text

    @Slot(object)
    def _ok(self, r):
        self.btn.setEnabled(True)
        from ..llm.backend import Message
        self._history.append(Message("user", self._pending))
        self._history.append(Message("assistant", r["text"]))
        warn = ""
        if not r["psel"]["clean"]:
            why = ", ".join(sorted({f['why'] for f in r['psel']['findings']}))
            warn = (f"<br><span style='color:#c33'>PSE-L upozorenje: {why}</span>")
        self.view.append(f"<b>{r['provider']}/{r['model']}</b> "
                         f"<i>({r['role']}, {r['routing_reason']})</i>{warn}<br>"
                         f"{r['text']}<br>")
        self._refresh_route()

    @Slot(str)
    def _fail(self, msg: str):
        self.btn.setEnabled(True)
        self.view.append(f"<span style='color:#c33'>{msg.splitlines()[0]}</span>")


# --------------------------------------------------------------------------- #
class ReportTab(QWidget):
    """The deliverable: every claim, with what backs it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        lay.addWidget(self.view, 1)
        btns = QHBoxLayout()
        b1 = QPushButton(t("generate"))
        b1.clicked.connect(self.generate)
        b2 = QPushButton(t("save"))
        b2.clicked.connect(self.save)
        btns.addWidget(b1)
        btns.addWidget(b2)
        btns.addStretch(1)
        lay.addLayout(btns)

    def build(self) -> str:
        store = get_store()
        counts = store.counts()
        out = ["# PSE J-Lab — izvestaj", "",
               f"generisano: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
               "## Stanje trezora", ""]
        for k in CardKind.ALL:
            out.append(f"- {k}: {counts.get(k, 0)}")
        j = journal.get().verify()
        out += ["", f"- lanac dnevnika: {'ispravan' if j['ok'] else 'PREKINUT'} "
                    f"({j['checked']} zapisa)", "", "## Izracunato (COMPUTED)", ""]
        for c in store.list(kind=CardKind.COMPUTED, limit=500):
            v = verify_run(c.run_id) if c.run_id else {"ok": False}
            out += [f"### {c.claim}",
                    f"- run: `{c.run_id}`",
                    f"- sha256: `{c.sha256}`",
                    f"- hes provera: {'poklapa se' if v.get('ok') else 'NE POKLAPA SE'}",
                    f"- {c.body}" if c.body else "", ""]
        for kind, title in ((CardKind.CITED, "Citirano (CITED)"),
                            (CardKind.SUSPECT, "Ranjeno (SUSPECT)"),
                            (CardKind.REFUTED, "Oboreno (REFUTED)"),
                            (CardKind.FOLKLORE, "Folklor (FOLKLORE)")):
            cards = store.list(kind=kind, limit=500)
            if not cards:
                continue
            out += [f"## {title}", ""]
            for c in cards:
                line = f"- {c.claim}"
                if c.reference:
                    line += f" — `{c.reference}`"
                out.append(line)
                for a in c.attacks:
                    out.append(f"  - [{a.get('verdict')}] {a.get('by')}: {a.get('argument')}")
            out.append("")
        out += ["## Napomena", "",
                "Svaka COMPUTED tvrdnja pokazuje na direktorijum sa rezultatom i njegov",
                "sha256. Ako se hes ne poklapa, rezultat je menjan posle pecacenja i",
                "tvrdnja vise ne vazi. Oborene kartice se nikada ne brisu."]
        return "\n".join(out)

    @Slot()
    def generate(self):
        self.view.setPlainText(self.build())

    @Slot()
    def save(self):
        text = self.view.toPlainText() or self.build()
        path, _ = QFileDialog.getSaveFileName(
            self, t("save"), str(Path.home() / "pse-jlab-izvestaj.md"),
            "Markdown (*.md)")
        if path:
            Path(path).write_text(text, encoding="utf-8")
            QMessageBox.information(self, t("save"), path)
