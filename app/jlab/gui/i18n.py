"""Two languages, one dictionary.  Serbian is the primary; English is complete."""

from __future__ import annotations

from ..core import config

STRINGS = {
    "app_title": ("PSE J-Lab — Jakobijani u masinskom ucenju",
                  "PSE J-Lab — Jacobians in machine learning"),
    "tab_home": ("Plan", "Plan"),
    "tab_lab": ("Laboratorija", "Laboratory"),
    "tab_runs": ("Rezultati", "Results"),
    "tab_vault": ("Trezor", "Vault"),
    "tab_refs": ("Reference", "References"),
    "tab_guardian": ("Guardian", "Guardian"),
    "tab_chat": ("Razgovor", "Chat"),
    "tab_report": ("Izvestaj", "Report"),
    "run": ("Pokreni", "Run"),
    "stop": ("Prekini", "Stop"),
    "params": ("Parametri", "Parameters"),
    "defaults": ("Vrati podrazumevano", "Reset to defaults"),
    "experiment": ("Eksperiment", "Experiment"),
    "running": ("radi...", "running..."),
    "done": ("gotovo", "done"),
    "failed": ("neuspelo", "failed"),
    "verify": ("Proveri", "Verify"),
    "make_card": ("Napravi karticu", "Make a card"),
    "claim": ("Tvrdnja", "Claim"),
    "kind": ("Vrsta", "Kind"),
    "topic": ("Tema", "Topic"),
    "search": ("Pretraga", "Search"),
    "wound": ("Rani (SUSPECT)", "Wound (SUSPECT)"),
    "refute": ("Obori (REFUTED)", "Refute"),
    "argument": ("Argument", "Argument"),
    "verify_all": ("Proveri sve reference", "Verify all references"),
    "admit_verified": ("Upisi potvrdjene kao kartice", "Admit verified as cards"),
    "send": ("Posalji", "Send"),
    "role": ("Uloga", "Role"),
    "generate": ("Napravi izvestaj", "Generate report"),
    "save": ("Sacuvaj", "Save"),
    "theta": ("theta", "theta"),
    "spent": ("potroseno", "spent"),
    "offline": ("bez modela (offline)", "no model (offline)"),
    "no_key": ("nijedan API kljuc nije podesen", "no API key configured"),
    "datasets_ok": ("podaci provereni", "datasets verified"),
    "datasets_bad": ("PODACI NE VALJAJU", "DATASET PROBLEM"),
    "wizard_title": ("Prvo pokretanje", "First run"),
    "wizard_intro": (
        "Pokreni ovo prvi put KOD KUCE, na mrezi kojoj verujes.\n"
        "Guardian tada uci TLS otiske svakog servera. Ako ih nauci na tudjem\n"
        "wifiju, naucice ono sto mu ta mreza pokaze.",
        "Run this for the first time AT HOME, on a network you trust.\n"
        "The Guardian learns each server's TLS fingerprint now. If it learns\n"
        "them on someone else's wifi, it learns whatever that network shows it."),
    "wizard_keys": ("API kljucevi (svi su opcioni — labovi rade bez mreze)",
                    "API keys (all optional — the labs need no network)"),
    "wizard_learn": ("Nauci TLS otiske sada", "Learn TLS pins now"),
    "wizard_finish": ("Zavrsi", "Finish"),
    "elapsed": ("trajanje", "elapsed"),
    "claim_hint": ("Tvrdnja mora imati bar 8 znakova i mora biti proverljiva.",
                   "A claim needs at least 8 characters and must be checkable."),
}


def t(key: str, lang: str | None = None) -> str:
    if lang is None:
        lang = config.load_settings().get("lang", "sr")
    pair = STRINGS.get(key)
    if not pair:
        return key
    return pair[0] if lang == "sr" else pair[1]
