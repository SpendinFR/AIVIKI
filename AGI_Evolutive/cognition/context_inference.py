from typing import Any, Dict, Tuple
import os
import platform
import time


def detect_runtime() -> Dict[str, Any]:
    rt = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "time": time.time(),
    }
    return {"runtime": rt, "score": 0.9, "contradictions": 0}


def detect_workspace(paths_hint: Dict[str, str] | None = None) -> Dict[str, Any]:
    ws: Dict[str, Any] = {}
    try:
        cwd = os.getcwd()
        ws["root"] = cwd
        ws["has_git"] = os.path.exists(os.path.join(cwd, ".git"))
        ws["files"] = [
            fn
            for fn in os.listdir(cwd)[:200]
            if fn.endswith((".py", ".md", ".json", ".yml", ".yaml", ".toml", ".ts", ".js", ".ipynb"))
        ]
        if paths_hint:
            ws["paths_hint"] = {k: v for k, v in paths_hint.items() if v}
    except Exception:
        pass
    score = 0.7 + 0.2 * (1.0 if ws.get("has_git") else 0.0)
    return {"workspace": ws, "score": min(0.95, score), "contradictions": 0}


def _memory_recent_interactions(arch, limit: int) -> list[Dict[str, Any]]:
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
    if not isinstance(data, list):
        return []
    return [m for m in data if m.get("kind") in {"interaction", "user", "message"}]


def infer_user_context(arch, lookback: int = 50) -> Dict[str, Any]:
    """
    Déduit langue/style/préfs observées à partir des interactions récentes.
    """
    lang = "fr"
    style: Dict[str, Any] = {}
    try:
        msgs = _memory_recent_interactions(arch, lookback)
        fr_hits = 0
        for msg in msgs:
            text = (msg.get("summary") or msg.get("text") or "").lower()
            if any(ch in text for ch in "éàèùçôâîû"):
                fr_hits += 1
            if "résume" in text or "plus court" in text:
                style["conciseness"] = "high"
            if "plus détaillé" in text or "explique davantage" in text:
                style["conciseness"] = "low"
            if "en français" in text:
                style["lang"] = "fr"
            if "english" in text and "please" in text:
                style["lang"] = "en"
        if msgs and fr_hits < (len(msgs) / 3):
            lang = "en"
    except Exception:
        pass
    ctx = {"user_lang": style.get("lang", lang), "style_hint": style}
    return {"context": ctx, "score": 0.7, "contradictions": 0}


def situational_summary(where: Dict[str, Any]) -> Tuple[str, float]:
    """
    Concatène un petit résumé et renvoie (texte, score global).
    """
    rt = where.get("runtime", {})
    ws = where.get("workspace", {})
    ctx = where.get("context", {})
    parts = []
    if rt:
        parts.append(
            f"Environnement: {rt.get('os')} {rt.get('os_release')} / Python {rt.get('python')} sur {rt.get('hostname')}."
        )
    if ws:
        root = ws.get("root")
        if root:
            parts.append(f"Espace de travail: {os.path.basename(root)} (git={ws.get('has_git')}).")
    if ctx:
        parts.append(
            f"Contexte utilisateur: langue={ctx.get('user_lang')}, style_hint={ctx.get('style_hint')}."
        )
    summary = " ".join(parts) or "Contexte en cours d'inférence."
    sub_scores = [
        where.get("runtime_score"),
        where.get("workspace_score"),
        where.get("context_score"),
    ]
    sub = [float(s) for s in sub_scores if isinstance(s, (int, float))]
    score = sum(sub) / len(sub) if sub else 0.75
    return summary, score


def infer_where_and_apply(arch, threshold: float = 0.70, stable_cycles: int = 2) -> Dict[str, Any]:
    """
    Infère runtime/workspace/context. Si score>=threshold et stable N cycles, écrit dans identity.where.*.
    La stabilité est estimée via un cache léger sur l'orchestrateur.
    """
    rt = detect_runtime()
    paths_hint = getattr(getattr(arch, "job_manager", None), "paths", None)
    ws = detect_workspace(paths_hint)
    uc = infer_user_context(arch)

    where = {
        "runtime": rt.get("runtime"),
        "runtime_score": rt.get("score"),
        "workspace": ws.get("workspace"),
        "workspace_score": ws.get("score"),
        "context": uc.get("context"),
        "context_score": uc.get("score"),
    }
    summary, global_score = situational_summary(where)
    where["summary"] = summary
    where["global_score"] = global_score

    last = getattr(arch, "_where_last", None)
    stable = False
    try:
        if isinstance(last, dict):
            previous = last.get("where")
            stamp = float(last.get("stamp", 0.0))
            if previous == where and (time.time() - stamp) >= 2.0:
                hits = int(last.get("hits", 0)) + 1
            else:
                hits = 1
        else:
            hits = 1
    except Exception:
        hits = 1
    stable = hits >= stable_cycles
    arch._where_last = {"stamp": time.time(), "where": where, "hits": hits}

    if global_score >= threshold and stable:
        try:
            arch.self_model.set_identity_patch(
                {
                    "where": {
                        "runtime": where["runtime"],
                        "workspace": where["workspace"],
                        "context": where["context"],
                        "summary": where["summary"],
                    }
                }
            )
            return {"status": "applied", "score": global_score, "summary": summary}
        except Exception:
            return {"status": "error", "score": global_score, "summary": summary}
    return {"status": "pending", "score": global_score, "summary": summary}
