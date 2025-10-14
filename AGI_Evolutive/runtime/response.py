from __future__ import annotations

from typing import Any, Dict, List


def ensure_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures required keys exist and normalises values."""
    defaults = {
        "hypothese_choisie": "clarifier intention",
        "incertitude": 0.5,
        "prochain_test": "-",
        "appris": [],
        "besoin": [],
    }
    data = {**defaults, **contract}
    data["appris"] = list(data.get("appris", []))
    data["besoin"] = list(data.get("besoin", []))
    data["incertitude"] = float(max(0.0, min(1.0, data.get("incertitude", 0.5))))
    return data


def format_agent_reply(base_text: str, **contract: Any) -> str:
    """Formats the agent reply mixing base text and contract clauses."""
    lines: List[str] = [base_text, ""]
    lines.append(f"🧭 Hypothèse: {contract.get('hypothese_choisie')}")
    lines.append(f"⚖️ Incertitude: {contract.get('incertitude'):.2f}")
    lines.append(f"🧪 Prochain test: {contract.get('prochain_test')}")
    if contract.get("appris"):
        appris = " | ".join(contract["appris"])
        lines.append(f"📚 J'apprends: {appris}")
    if contract.get("besoin"):
        besoin = " | ".join(contract["besoin"])
        lines.append(f"🤝 Besoin: {besoin}")
    return "\n".join(lines)
from typing import Any, Dict, List, Optional

CONTRACT_KEYS = [
    "hypothese_choisie",
    "incertitude",
    "prochain_test",
    "appris",
    "besoin",
]


def _stringify_list(items: Optional[List[str]]) -> str:
    if not items:
        return "-"
    return "\n".join([f"• {x}" for x in items])


def format_agent_reply(
    base_text: str,
    *,
    hypothese_choisie: str,
    incertitude: float,
    prochain_test: Optional[str],
    appris: Optional[List[str]] = None,
    besoin: Optional[List[str]] = None,
) -> str:
    """
    Formate TOUTES les réponses pour éviter le générique.
    """
    if incertitude < 0:
        incertitude = 0.0
    if incertitude > 1:
        incertitude = 1.0

    learned = _stringify_list(appris)
    needs = _stringify_list(besoin)
    test_line = prochain_test or "-"

    return (
        f"{base_text}\n\n"
        f"-\n"
        f"🧩 Hypothèse prise: {hypothese_choisie}\n"
        f"🤔 Incertitude: {incertitude:.2f}\n"
        f"🧪 Prochain test: {test_line}\n"
        f"📗 Ce que j'apprends: \n{learned}\n"
        f"🔧 Besoins: \n{needs}"
    )


def ensure_contract(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Complète les champs manquants de contrat si besoin."""
    out = dict(kwargs)
    out.setdefault(
        "hypothese_choisie", "clarifier l'intention et la granularité attendue"
    )
    out.setdefault("incertitude", 0.5)
    out.setdefault(
        "prochain_test", "proposer 2 chemins d'action et demander ton choix"
    )
    out.setdefault("appris", ["prioriser le concret et la traçabilité"])
    out.setdefault(
        "besoin", ["confirmer si tu préfères plan en étapes ou patch direct"]
    )
    return out
