from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from AGI_Evolutive.core.structures.mai import Bid, MAI
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore

# --- MAI dispatcher (NLG) ---
# Registre de handlers extensible à chaud (aucune liste figée)
_MAI_NLG_HANDLERS: Dict[str, Callable[[Bid, "NLGContext"], None]] = {}


def register_nlg_handler(action_hint: str, fn: Callable[[Bid, "NLGContext"], None]) -> None:
    """Appelé lors de la promotion d’un MAI si celui-ci fournit un handler dédié."""

    _MAI_NLG_HANDLERS[action_hint] = fn


class NLGContext:
    """Context runtime minimaliste pour appliquer des bids MAI au rendu NLG."""

    def __init__(self, base_text: str, apply_hint: Callable[[str, str], str]):
        self.text = base_text
        self._apply_hint = apply_hint
        self._applied: List[Dict[str, str]] = []

    # --- API utilisée par les handlers ---
    def mark_applied(self, bid: Bid) -> None:
        entry = {"origin": bid.source, "hint": bid.action_hint}
        if entry not in self._applied:
            self._applied.append(entry)

    def register_custom_action(self, origin: str, hint: str) -> None:
        entry = {"origin": origin, "hint": hint}
        if entry not in self._applied:
            self._applied.append(entry)

    def apply_bid_hint(self, bid: Bid) -> None:
        hint = (bid.action_hint or "").strip()
        self.text = self._apply_hint(self.text, hint)
        self.mark_applied(bid)

    def redact(self, fields: Any, *, bid: Optional[Bid] = None) -> None:
        if not fields:
            return
        if isinstance(fields, (set, list, tuple)):
            payload = ", ".join(sorted(str(x) for x in fields if x is not None))
        else:
            payload = str(fields)
        note = f"(Je masque {payload} pour protéger la confidentialité.)"
        current = (self.text or "").strip()
        if note.lower() not in current.lower():
            self.text = f"{current}\n\n{note}" if current else note
        if bid is not None:
            self.mark_applied(bid)

    def rephrase_politely(self, *, bid: Optional[Bid] = None) -> None:
        self.text = self._apply_hint(self.text, "RephraseRespectfully")
        if bid is not None:
            self.mark_applied(bid)

    def applied_hints(self) -> List[Dict[str, str]]:
        return list(self._applied)


def apply_generic(bid: Bid, nlg_context: "NLGContext") -> None:
    """Fallback générique basé sur le contenu d’un Bid."""

    if isinstance(bid.target, dict) and "redact" in bid.target:
        nlg_context.redact(bid.target["redact"], bid=bid)
    elif (bid.action_hint or "").lower().startswith("rephrase"):
        nlg_context.rephrase_politely(bid=bid)
    else:
        nlg_context.apply_bid_hint(bid)


def apply_mai_bids_to_nlg(
    nlg_context: "NLGContext",
    state: Optional[Dict[str, Any]],
    predicate_registry: Optional[Dict[str, Any]],
) -> List[MAI]:
    ms = MechanismStore()
    try:
        mechanisms: Iterable[MAI] = ms.scan_applicable(state or {}, predicate_registry or {})
    except Exception:
        mechanisms = []

    applied: List[MAI] = []
    for mechanism in mechanisms:
        applied.append(mechanism)
        try:
            bids = mechanism.propose(state or {})
        except Exception:
            continue
        for bid in bids or []:
            handler = _MAI_NLG_HANDLERS.get(bid.action_hint)
            if handler:
                try:
                    handler(bid, nlg_context)
                except Exception:
                    continue
                nlg_context.mark_applied(bid)
            else:
                apply_generic(bid, nlg_context)
    return applied
