from typing import Any, Dict


def _memory_recent(arch, limit: int):
    store = None
    memory = getattr(arch, "memory", None)
    if memory is not None:
        store = getattr(memory, "store", None)
    if store is None:
        store = getattr(arch, "_memory_store", None)
    if store is None or not hasattr(store, "get_recent"):
        return []
    try:
        data = store.get_recent(limit)
    except Exception:
        return []
    return list(data) if isinstance(data, list) else []


def infer_preferences(arch, window: int = 100) -> Dict[str, Any]:
    """
    Observe des corrections/choix récents pour déduire values/likes/dislikes/style.
    Retourne un patch + score de confiance (0..1).
    """
    likes, dislikes, values = set(), set(), set()
    style: Dict[str, Any] = {}

    events = _memory_recent(arch, window)
    for ev in reversed(events):
        text = str(ev.get("text") or ev.get("summary") or "").lower()
        if "plus court" in text or "résume" in text:
            style["conciseness"] = "high"
        if "plus détaillé" in text or "explique davantage" in text:
            style["conciseness"] = "low"
        if "en français" in text:
            style["lang"] = "fr"
        if "english" in text and "please" in text:
            style["lang"] = "en"
        if "sources" in text or "citer" in text:
            values.add("traceability")
        kind = str(ev.get("kind"))
        if kind == "decision":
            action = str(ev.get("action") or "").lower()
            if "safeguard" in action or "redact" in action:
                values.add("care")
            if "explore" in action or "hypotheses" in action:
                values.add("curiosity")

    signals = len(values) + len(style)
    score = min(1.0, 0.5 + 0.1 * signals)
    patch: Dict[str, Any] = {
        "preferences": {},
    }
    if values:
        patch["preferences"]["values"] = sorted(values)
    if likes:
        patch["preferences"]["likes"] = sorted(likes)
    if dislikes:
        patch["preferences"]["dislikes"] = sorted(dislikes)
    if style:
        patch["preferences"]["style"] = style

    if not patch["preferences"]:
        return {"patch": {"preferences": {}}, "score": score}
    return {"patch": patch, "score": score}


def apply_preferences_if_confident(arch, threshold: float = 0.75) -> Dict[str, Any]:
    res = infer_preferences(arch)
    patch = res.get("patch", {"preferences": {}})
    score = float(res.get("score", 0.0))
    preferences = patch.get("preferences", {}) if isinstance(patch, dict) else {}
    if not preferences:
        return {"status": "no_change", "score": score}
    if score >= threshold:
        try:
            arch.self_model.set_identity_patch(patch)
            return {"status": "applied", "score": score, "patch": patch}
        except Exception:
            return {"status": "error", "score": score, "patch": patch}
    return {"status": "needs_confirmation", "score": score, "patch": patch}
