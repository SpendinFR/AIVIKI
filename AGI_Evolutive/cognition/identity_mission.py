from typing import Any, Dict, List, Tuple
import math
import time


def _norm(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_policy(arch) -> Any:
    if arch is None:
        return None
    if hasattr(arch, "policy"):
        return arch.policy
    core = getattr(arch, "core", None)
    if core is not None and hasattr(core, "policy"):
        return core.policy
    return getattr(arch, "_policy_engine", None)


def _memory_recent(arch, limit: int) -> List[Dict[str, Any]]:
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
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def mine_frequent_goals(arch, horizon: int = 500) -> Dict[str, Any]:
    """
    Analyse les jobs/decisions récents pour extraire les thèmes dominants.
    Retourne {"topics":[(topic, coverage_score), ...], "evidence_refs":[...]}
    """
    topics: Dict[str, int] = {}
    refs: List[str] = []

    jm = getattr(arch, "job_manager", None)
    if jm and hasattr(jm, "snapshot_identity_view"):
        try:
            view = jm.snapshot_identity_view()
        except Exception:
            view = {}
        recent_jobs = (view.get("recent") or [])[:horizon]
        for r in recent_jobs:
            topic = str(r.get("kind") or r.get("topic") or "__unknown__")
            topics[topic] = topics.get(topic, 0) + 1
            if r.get("job_id"):
                refs.append(f"job:{r['job_id']}")

    recent_memories = list(reversed(_memory_recent(arch, min(200, horizon))))
    count = 0
    for entry in recent_memories:
        if count >= horizon:
            break
        if str(entry.get("kind")) == "decision":
            topic = entry.get("topic") or "__generic__"
            topics[topic] = topics.get(topic, 0) + 1
            if entry.get("decision_id"):
                refs.append(f"decision:{entry['decision_id']}")
            count += 1

    total = sum(topics.values()) or 1
    ranked = sorted(((k, v / total) for k, v in topics.items()), key=lambda kv: kv[1], reverse=True)
    return {"topics": ranked[:10], "evidence_refs": refs[:200]}


def cluster_intent_constraints(arch, top_k: int = 5) -> Dict[str, Any]:
    """
    Interroge le modèle d'intent/contraintes pour obtenir les contraintes récurrentes.
    Retourne {"constraints":[("desc", weight), ...]}
    """
    lst: List[Tuple[str, float]] = []
    im = getattr(arch, "intent_model", None)
    if im and hasattr(im, "constraints_view"):
        try:
            view = im.constraints_view(top_k=top_k) or []
        except Exception:
            view = []
        for c in view:
            if not isinstance(c, dict):
                continue
            desc = c.get("description") or c.get("name") or ""
            if not desc:
                continue
            w = float(c.get("weight", 0.5))
            lst.append((desc, _norm(w)))
    return {"constraints": lst}


def draft_mission_hypotheses(freq: Dict[str, Any], cons: Dict[str, Any], feedback_score: float) -> Dict[str, Any]:
    """
    Produit 2–3 formulations candidates avec score:
      score = 0.5*coverage (top topics) + 0.3*constraint_overlap + 0.2*feedback_score
    """
    topics = [t for t, _ in (freq.get("topics") or [])[:3]]
    cons_desc = [d for d, _ in (cons.get("constraints") or [])[:3]]

    candidates: List[str] = []
    if topics:
        joined_topics = ", ".join(topics)
        guard = ", ".join(cons_desc) if cons_desc else "les contraintes usuelles"
        candidates.append(f"Aider efficacement sur {joined_topics} en respectant {guard}.")
    hyp2 = "Maximiser la valeur des échanges en apprenant en continu et en garantissant la sécurité des informations."
    if hyp2 not in candidates:
        candidates.append(hyp2)
    if cons_desc:
        hyp3 = f"Prioriser {cons_desc[0]} tout en améliorant la précision et la clarté des réponses."
        if hyp3 not in candidates:
            candidates.append(hyp3)

    cov = sum(w for _, w in (freq.get("topics") or [])[:3])
    cov = _norm(cov)
    con = 0.0
    if cons.get("constraints"):
        con = sum(w for _, w in cons["constraints"][:3]) / 3.0
    con = _norm(con)
    fb = _norm(feedback_score)

    scored: List[Tuple[str, float]] = []
    for h in candidates[:3]:
        score = 0.5 * cov + 0.3 * con + 0.2 * fb
        scored.append((h, _norm(score)))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return {"candidates": scored}


def recommend_and_apply_mission(arch, threshold: float = 0.75, delta_gate: float = 0.10) -> Dict[str, Any]:
    """
    Chaîne complète: mine -> cluster -> draft -> décider d'écrire.
    Ecrit via Policy.apply_proposal si score >= threshold ET (best - second) >= delta_gate,
    sinon retourne une recommandation (l'orchestrateur pourra déclencher QM si blocage).
    """
    freq = mine_frequent_goals(arch)
    cons = cluster_intent_constraints(arch)

    fb = 0.6
    policy = _resolve_policy(arch)
    try:
        stats = getattr(policy, "stats", None)
        if isinstance(stats, dict):
            success = float(stats.get("success", 0))
            fail = float(stats.get("fail", 0))
            total = success + fail
            if total > 0:
                fb = _norm(success / total)
    except Exception:
        pass

    draft = draft_mission_hypotheses(freq, cons, fb)
    candidates = draft.get("candidates", [])
    if not candidates:
        return {"status": "no_candidates", "candidates": []}

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else (None, 0.0)
    delta = best[1] - (second[1] if second else 0.0)

    decided = False
    if best[1] >= threshold and delta >= delta_gate:
        proposal = {
            "type": "update",
            "path": ["identity", "purpose", "mission"],
            "value": best[0],
            "rationale": "Mission inferred from frequent goals, constraints and feedback.",
            "evidence_refs": freq.get("evidence_refs", [])[:50],
        }
        try:
            if hasattr(arch, "self_model") and policy is not None:
                arch.self_model.apply_proposal(proposal, policy)
                decided = True
        except Exception:
            decided = False

    return {
        "status": "applied" if decided else "needs_confirmation",
        "best": best,
        "second": second,
        "delta": delta,
        "freq": freq,
        "constraints": cons,
    }
