from __future__ import annotations

from typing import Any, Dict, List, Optional

CONTRACT_KEYS = [
    "hypothese_choisie",
    "incertitude",
    "prochain_test",
    "appris",
    "besoin",
]

CONTRACT_DEFAULTS: Dict[str, Any] = {
    "hypothese_choisie": "clarifier l'intention et la granularité attendue",
    "incertitude": 0.5,
    "prochain_test": "proposer 2 chemins d'action et demander ton choix",
    "appris": ["prioriser le concret et la traçabilité"],
    "besoin": ["confirmer si tu préfères plan en étapes ou patch direct"],
}


def _stringify_list(items: Optional[List[str]]) -> str:
    if not items:
        return "-"
    return "\n".join([f"• {x}" for x in items])


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, (set, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def ensure_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults with provided values and normalise the payload."""

    normalised: Dict[str, Any] = dict(CONTRACT_DEFAULTS)
    normalised.update(contract or {})

    normalised["appris"] = _ensure_list(normalised.get("appris"))
    normalised["besoin"] = _ensure_list(normalised.get("besoin"))

    try:
        incertitude = float(normalised.get("incertitude", 0.5))
    except (TypeError, ValueError):
        incertitude = 0.5
    normalised["incertitude"] = max(0.0, min(1.0, incertitude))

    if not normalised.get("prochain_test"):
        normalised["prochain_test"] = "-"

    return normalised


def format_agent_reply(base_text: str, **contract: Any) -> str:
    """Formats an agent reply mixing the base text with the social contract."""

    enriched = ensure_contract(contract)

    learned = _stringify_list(enriched.get("appris"))
    needs = _stringify_list(enriched.get("besoin"))
    test_line = enriched.get("prochain_test") or "-"

    return (
        f"{base_text}\n\n"
        f"-\n"
        f"🧩 Hypothèse prise: {enriched['hypothese_choisie']}\n"
        f"🤔 Incertitude: {enriched['incertitude']:.2f}\n"
        f"🧪 Prochain test: {test_line}\n"
        f"📗 Ce que j'apprends: \n{learned}\n"
        f"🔧 Besoins: \n{needs}"
    )
