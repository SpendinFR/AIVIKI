from typing import Any, Dict, List


def _resolve_policy(arch) -> Any:
    if arch is None:
        return None
    if hasattr(arch, "policy"):
        return arch.policy
    core = getattr(arch, "core", None)
    if core is not None and hasattr(core, "policy"):
        return core.policy
    return getattr(arch, "_policy_engine", None)


def extract_effective_policies(arch) -> List[Dict[str, Any]]:
    """
    Inspecte la Policy pour lister des règles / comportements déjà effectifs.
    Retourne [{key, desc, strength}, ...]
    """
    out: List[Dict[str, Any]] = []
    pol = _resolve_policy(arch)
    if not pol:
        return out
    for key in ["abstention_threshold", "max_depth", "risk_aversion", "privacy_mode"]:
        if hasattr(pol, key):
            val = getattr(pol, key)
            out.append({"key": key, "desc": f"{key}={val}", "strength": 0.6})
    try:
        stats = getattr(pol, "stats", None)
        if isinstance(stats, dict):
            success = stats.get("success", 0)
            fail = stats.get("fail", 0)
            out.append(
                {
                    "key": "consistency",
                    "desc": f"success={success}/fail={fail}",
                    "strength": 0.5,
                }
            )
    except Exception:
        pass
    return out


def map_to_principles(effective: List[Dict[str, Any]], values: List[str]) -> List[Dict[str, Any]]:
    """
    Convertit des règles effectives et des valeurs en étiquette de principes {key, desc}.
    """
    pr: List[Dict[str, Any]] = []
    low = [v.lower() for v in (values or [])]
    joined_values = " ".join(low)
    if "care" in low or "sécurité" in joined_values:
        pr.append({"key": "do_no_harm", "desc": "Minimiser les risques, préserver la confidentialité."})
    if "curiosity" in low:
        pr.append({"key": "keep_learning", "desc": "Apprendre en continu à partir des retours."})
    pr.append({"key": "honesty", "desc": "Dire quand l'incertitude est élevée et citer les limites."})
    for e in effective:
        if e.get("key") == "privacy_mode":
            pr.append({"key": "respect_privacy", "desc": "Ne pas divulguer d'informations sensibles."})
        if e.get("key") == "risk_aversion":
            pr.append({"key": "prudence", "desc": "Préférer les actions réversibles quand l'incertitude est haute."})
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in pr:
        dedup.setdefault(item["key"], item)
    return list(dedup.values())


def propose_commitments(principles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Propose des engagements booléens à activer dans SelfModel.identity.commitments.by_key
    """
    props: List[Dict[str, Any]] = []
    for p in principles:
        if p.get("key") == "respect_privacy":
            props.append(
                {
                    "key": "respect_privacy",
                    "active": True,
                    "note": "Explicit user-endorsed privacy commitment.",
                }
            )
        if p.get("key") == "honesty":
            props.append(
                {
                    "key": "disclose_uncertainty",
                    "active": True,
                    "note": "State uncertainty when confidence < threshold.",
                }
            )
    return props


def run_and_apply_principles(arch, require_confirmation: bool = True) -> Dict[str, Any]:
    """
    Chaîne complète:
      - extrait règles effectives + valeurs
      - mappe en principes
      - propose des engagements
      - écrit (Policy.apply_proposal) si pas sensible, sinon retourne 'needs_confirmation'
    """
    if not hasattr(arch, "self_model"):
        return {"applied": [], "pending": [], "principles": [], "commitments": []}

    self_model = arch.self_model
    self_model.ensure_identity_paths()
    identity = getattr(self_model, "identity", {}) or {}
    values = identity.get("preferences", {}).get("values", []) if isinstance(identity, dict) else []

    eff = extract_effective_policies(arch)
    prin = map_to_principles(eff, values)
    commits = propose_commitments(prin)

    applied: List[str] = []
    pending: List[str] = []

    if prin:
        proposal = {
            "type": "update",
            "path": ["identity", "principles"],
            "value": prin,
            "rationale": "Derived from effective Policy and current values.",
        }
        policy = _resolve_policy(arch)
        try:
            if policy is not None:
                self_model.apply_proposal(proposal, policy)
                applied.append("principles")
            else:
                pending.append("principles")
        except Exception:
            pending.append("principles")

    for cm in commits:
        key = cm.get("key")
        if not key:
            continue
        label = f"commit:{key}"
        if require_confirmation:
            pending.append(label)
            continue
        try:
            self_model.set_commitment(key, bool(cm.get("active", True)), note=cm.get("note", ""))
            applied.append(label)
        except Exception:
            pending.append(label)

    return {"applied": applied, "pending": pending, "principles": prin, "commitments": commits}
