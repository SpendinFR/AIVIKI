from __future__ import annotations

from typing import Any, Dict, List


def ensure_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures required keys exist and normalises values."""
    defaults = {
        "hypothese_choisie": "clarifier intention",
        "incertitude": 0.5,
        "prochain_test": "—",
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
