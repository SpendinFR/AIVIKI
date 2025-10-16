from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

Number = float


def _now() -> float:
    return time.time()


def _safe(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _tok(s: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ]{3,}", (s or "").lower())


class GoalPrioritizer:
    """
    Prioritiseur multi-signaux.
    - Lit persona/values (+ alias & poids si dispo), ontology/beliefs, skills appris,
      homeostasis.drives, questions/uncertainty, deadlines, dépendances parent↔enfant,
      progrès/ancienneté, directives utilisateur récentes, coût/risque estimé des actions,
      et avis de Policy.
    - Produit: plan["priority"] ∈ [0..1] + plan["tags"] (ex: ["urgent"] ou ["background"]).
    - N'enlève rien: si un composant manque, il s'efface (fallback = 0).
    - Configurable par JSON: data/prioritizer_config.json (facultatif).
    """

    def __init__(self, arch):
        self.arch = arch
        self.cfg = self._load_cfg()
        # petite mémoire interne pour anti-famine et “dernier tick”
        self._last_seen: Dict[str, float] = {}
        self._lane_thresholds = self.cfg.get(
            "lane_thresholds",
            {
                "urgent_pr": 0.92,
                "background_pr": 0.60,
            },
        )

    # ---------- CONFIG ----------
    def _load_cfg(self) -> Dict[str, Any]:
        path = getattr(self.arch, "prioritizer_cfg_path", "data/prioritizer_config.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        # défaut raisonnable
        return {
            "weights": {
                "user_urgency": 0.30,
                "deadline": 0.28,
                "uncertainty": 0.18,
                "drive_alignment": 0.14,
                "identity_alignment": 0.18,
                "parent_waiting": 0.10,
                "staleness": 0.10,
                "base_priority": 0.10,
                "policy_feasibility": 0.10,
                "competence_fit": 0.10,
                "novelty_drive": 0.08,
                "action_cost": -0.08,  # coût pénalise
                "risk_penalty": -0.10,  # risque pénalise
            },
            "lane_thresholds": {
                "urgent_pr": 0.92,
                "background_pr": 0.60,
            },
            "background_period_sec": 5 * 60,  # utile si tu filtres ailleurs la fréquence BG
        }

    # ---------- FEATURES (chacune retourne (score, reason)) ----------
    def feat_user_urgency(self) -> Tuple[Number, str]:
        # détecte "maintenant/urgent", directives NL récentes, etc.
        try:
            rec = self.arch.memory.get_recent_memories(limit=30)
        except Exception:
            rec = []
        for m in reversed(rec):
            if (m.get("kind") or "") != "interaction":
                continue
            if (m.get("role") or "") != "user":
                continue
            text = (m.get("text") or "").lower()
            if re.search(r"\b(urgent|prioritaire|tout\s+de\s+suite|maintenant)\b", text):
                return (1.0, "user_urgent")
            # directives style (ex: fais preuve d’empathie)
            if re.search(r"\bfais\s+preuve\s+d['e]?\w+", text):
                return (0.65, "user_directive")
            # demandes avec échéancier
            if re.search(r"\b(demain|aujourd'hui|ce\s+soir|avant)\b", text):
                return (0.5, "user_time_ref")
            break
        return (0.0, "")

    def feat_deadline(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        dl = plan.get("deadline_ts")
        if not dl:
            return (0.0, "")
        rem = dl - _now()
        if rem <= 0:
            return (1.0, "deadline_passed")
        # mapping: <=1h -> ~1.0 ; 48h -> ~0
        v = max(0.0, min(1.0, 1.0 - rem / (48 * 3600)))
        return (v, "deadline")

    def feat_uncertainty(self, goal_id: str) -> Tuple[Number, str]:
        # incertitude haute -> priorité à lever les blocages / clarifier
        u = 0.0
        reason = ""
        try:
            pol = getattr(self.arch, "policy", None)
            if pol and hasattr(pol, "confidence"):
                c = float(pol.confidence())
                u = max(u, 1.0 - c)
                if u > 0.5:
                    reason = "low_policy_conf"
        except Exception:
            pass
        try:
            qm = getattr(self.arch, "question_manager", None)
            if qm:
                for q in getattr(qm, "pending_questions", []):
                    # si question liée au goal_id, boost supplémentaire
                    if goal_id.lower() in (q.get("topic", "") or q.get("text", "")).lower():
                        u = max(u, 0.7)
                        reason = reason or "pending_question"
                        break
        except Exception:
            pass
        return (u, reason)

    def feat_drive_alignment(self, goal_id: str) -> Tuple[Number, str]:
        # aligne sur drives (homeostasis)
        drives = _safe(self.arch, "homeostasis", "state", "drives", default={}) or {}
        g = goal_id.lower()
        w = 0.0
        why: List[str] = []
        # mapping heuristique + extensible via ontology (voir feat_identity_alignment)
        if any(k in g for k in ("understand", "learn", "research", "investigate")):
            v = float(drives.get("curiosity", 0.5))
            w += 0.5 * v
            why.append(f"curiosity={v:.2f}")
        if any(k in g for k in ("human", "emotion", "empathy", "social")):
            v = float(drives.get("social_bonding", 0.5))
            w += 0.5 * v
            why.append(f"social_bonding={v:.2f}")
        if "self" in g or "evolve" in g:
            v = float(drives.get("self_actualization", drives.get("autonomy", 0.5)))
            w += 0.4 * v
            why.append(f"self_actualization={v:.2f}")
        return (min(1.0, w), ",".join(why))

    def _goal_concepts(self, goal_id: str, plan: Dict[str, Any]) -> List[str]:
        # 1) meta stockée
        cs = (plan.get("concepts") or []) + (plan.get("tags") or [])
        # 2) parsage id/title
        cs += [
            w
            for w in _tok(goal_id)
            if w not in ("understand", "learn", "goal", "task", "subgoal")
        ]
        title = plan.get("title") or ""
        cs += [w for w in _tok(title)]
        return list(dict.fromkeys(cs))[:10]

    def _value_aliases(self) -> Dict[str, List[str]]:
        # persona.values + alias éventuels
        persona = _safe(self.arch, "self_model", "state", "persona", default={}) or {}
        alias = persona.get("value_aliases") or {}
        vals = persona.get("values") or []
        for v in vals:
            alias.setdefault(v.lower(), [v.lower()])
        return alias

    def _ontology_neighbors(self, term: str, max_depth: int = 2) -> List[str]:
        # essaie via ontology (si présente), sinon retour vide
        out: List[str] = []
        onto = getattr(self.arch, "ontology", None)
        if not onto:
            return out
        try:
            nid = f"concept:{term}"
            if not onto.has_entity(nid):
                return out
            # neighbors 1..max_depth (API hypothétique -> adapte à la tienne)
            frontier: List[Tuple[str, int]] = [(nid, 0)]
            seen = {nid}
            while frontier:
                cur, d = frontier.pop(0)
                if d >= max_depth:
                    continue
                neigh: List[str] = []
                try:
                    neigh = list(onto.neighbors(cur))
                except Exception:
                    pass
                for nb in neigh:
                    if nb in seen:
                        continue
                    seen.add(nb)
                    out.append(nb)
                    frontier.append((nb, d + 1))
        except Exception:
            pass
        # dé-normalise "concept:xxx" -> "xxx"
        clean: List[str] = []
        for n in out:
            if isinstance(n, str) and n.startswith("concept:"):
                clean.append(n.split(":", 1)[1])
        return clean

    def feat_identity_alignment(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        """
        Score d'identité riche:
        - correspondance concepts du goal ↔ persona.values (avec alias)
        - expansion via ontologie jusqu'à 2 sauts (synonymes, proches)
        - support par beliefs (évidences récentes)
        - renforcement si la valeur est récente (feedback utilisateur, virtue_learned)
        """

        values_alias = self._value_aliases()
        if not values_alias:
            return (0.0, "")

        concepts = self._goal_concepts(goal_id, plan)
        # expand via ontology
        expanded = set(concepts)
        for c in concepts:
            for nb in self._ontology_neighbors(c, max_depth=2):
                expanded.add(nb)

        # matching alias
        score = 0.0
        matches: List[str] = []
        for val, aliases in values_alias.items():
            # Intersection entre alias de la valeur et concepts/voisins
            if set(aliases) & set(expanded):
                score += 0.35
                matches.append(val)

        # beliefs support (si un predicate 'promotes' lie la valeur au concept)
        beliefs = getattr(self.arch, "beliefs", None)
        if beliefs and hasattr(beliefs, "support"):
            try:
                for c in concepts[:5]:
                    s = beliefs.support(
                        subject=f"concept:{c}", predicate="promotes_identity", default=0.0
                    )
                    if s > 0:
                        score += min(0.25, s)
                        matches.append(f"belief:{c}")
            except Exception:
                pass

        # événements récents (virtue_learned, feedback) -> boost léger
        try:
            rec = self.arch.memory.get_recent_memories(limit=50)
            for m in rec[-50:]:
                if (m.get("kind") or "") == "virtue_learned":
                    v = (m.get("value") or "").lower()
                    if v in values_alias.keys():
                        score += 0.15
                        matches.append(f"virtue:{v}")
                if (m.get("kind") or "") == "feedback" and "style" in (m.get("tags") or []):
                    score += 0.05
        except Exception:
            pass

        return (min(1.0, score), ",".join(matches[:4]))

    def feat_parent_waiting(self, goal_id: str) -> Tuple[Number, str]:
        parent = _safe(self.arch, "planner", "state", "parents", default={}).get(goal_id)
        if not parent:
            return (0.0, "")
        parent_plan = _safe(self.arch, "planner", "state", "plans", default={}).get(parent, {})
        if parent_plan and parent_plan.get("status") != "done":
            return (0.4, f"parent:{parent}")
        return (0.0, "")

    def feat_staleness(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        last = plan.get("last_tick_done") or plan.get("created_ts") or (_now() - 1800)
        age = _now() - last
        # à 30 min sans progrès → +0.5 ; decay sinon
        v = max(0.0, min(0.5, age / (30 * 60)))
        return (v, f"stale:{int(age)}s")

    def feat_base_priority(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        bp = float(plan.get("priority", 0.5))
        return (bp, f"base:{bp:.2f}")

    def feat_policy_feasibility(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si la policy va probablement refuser la majorité des steps => baisse
        pol = getattr(self.arch, "policy", None)
        if not pol:
            return (0.0, "")
        # heuristique: si plan a beaucoup d'ops "restricted" (selon policy)
        ops: List[str] = []
        for st in plan.get("steps", []):
            if isinstance(st, dict) and st.get("kind") == "act":
                ops.append(st.get("op"))
        if not ops:
            return (0.0, "")
        bad = 0
        for op in ops[:6]:
            try:
                r = pol.simulate(op, plan) if hasattr(pol, "simulate") else None
                if isinstance(r, dict) and r.get("decision") == "deny":
                    bad += 1
            except Exception:
                pass
        if bad >= max(2, len(ops) // 2):
            return (-0.5, "policy_block_risk")
        return (0.0, "")

    def feat_competence_fit(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si on possède déjà les skills utiles, on livre plus vite → boost
        # si totalement inconnu mais curiosité haute → boost via novelty_drive plutôt
        skills: Dict[str, Any] = {}
        try:
            path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    skills = json.load(f) or {}
        except Exception:
            pass
        concepts = self._goal_concepts(goal_id, plan)
        have = sum(1 for c in concepts if c in skills and skills[c].get("acquired"))
        if have == 0:
            return (0.0, "")
        v = min(1.0, 0.15 * have)
        return (v, f"skills:{have}")

    def feat_novelty_drive(self, goal_id: str, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # curiosité + absence de skill => exploration
        drives = _safe(self.arch, "homeostasis", "state", "drives", default={}) or {}
        curiosity = float(drives.get("curiosity", 0.5))
        skills: Dict[str, Any] = {}
        try:
            path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    skills = json.load(f) or {}
        except Exception:
            pass
        concepts = self._goal_concepts(goal_id, plan)
        unknown = sum(1 for c in concepts if c not in skills)
        if unknown == 0:
            return (0.0, "")
        v = min(1.0, 0.1 * unknown * (0.6 + 0.7 * curiosity))
        return (v, f"novel:{unknown}")

    def feat_action_cost(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si on a des stats sur les ops (avg_ms), on pénalise légèrement les plans “chers”
        stats = getattr(self.arch, "actions", None)
        if not stats or not hasattr(stats, "stats"):
            return (0.0, "")
        total_ms = 0.0
        n = 0
        for st in plan.get("steps", []):
            if isinstance(st, dict) and st.get("kind") == "act":
                op = st.get("op")
                meta = stats.stats.get(op) if isinstance(stats.stats, dict) else None
                if meta and "avg_ms" in meta:
                    total_ms += float(meta["avg_ms"])
                    n += 1
        if n == 0:
            return (0.0, "")
        avg = total_ms / n
        # 0 si <30ms ; -0.2 vers -0.4 si >300ms
        pen = -min(0.4, max(0.0, (avg - 30.0) / 300.0))
        return (pen, f"cost_ms~{avg:.0f}")

    def feat_risk_penalty(self, plan: Dict[str, Any]) -> Tuple[Number, str]:
        # si le plan a un tag 'risky' ou des steps marqués 'needs_human' → pénalité (laisse Policy trancher)
        tags = set(plan.get("tags", []))
        if "risky" in tags:
            return (-0.3, "risky_tag")
        if any(
            (isinstance(st, dict) and st.get("policy") == "needs_human")
            for st in plan.get("steps", [])
        ):
            return (-0.15, "needs_human_steps")
        return (0.0, "")

    # ---------- SCORING GLOBAL ----------
    def score_goal(self, goal_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        W = self.cfg["weights"]
        parts: List[Tuple[str, float, str]] = []

        def add(name: str, val_reason: Tuple[Number, str]):
            v, r = val_reason
            parts.append((name, v, r))

        # calcule toutes les features (avec fallbacks)
        add("user_urgency", self.feat_user_urgency())
        add("deadline", self.feat_deadline(plan))
        add("uncertainty", self.feat_uncertainty(goal_id))
        add("drive_alignment", self.feat_drive_alignment(goal_id))
        add("identity_alignment", self.feat_identity_alignment(goal_id, plan))
        add("parent_waiting", self.feat_parent_waiting(goal_id))
        add("staleness", self.feat_staleness(goal_id, plan))
        add("base_priority", self.feat_base_priority(plan))
        add("policy_feasibility", self.feat_policy_feasibility(goal_id, plan))
        add("competence_fit", self.feat_competence_fit(goal_id, plan))
        add("novelty_drive", self.feat_novelty_drive(goal_id, plan))
        add("action_cost", self.feat_action_cost(plan))
        add("risk_penalty", self.feat_risk_penalty(plan))

        # somme pondérée
        pr = 0.0
        reasons: List[str] = []
        for name, v, r in parts:
            w = float(W.get(name, 0.0))
            pr += w * v
            if r:
                reasons.append(f"{name}:{r}({v:.2f}×{w:.2f})")

        pr = max(0.0, min(1.0, pr))

        # tags de voie (ne supprime pas les autres tags)
        tags = set(plan.get("tags", []))
        tags.discard("urgent")
        tags.discard("background")
        if pr >= self._lane_thresholds["urgent_pr"]:
            tags.add("urgent")
        elif pr < self._lane_thresholds["background_pr"]:
            tags.add("background")

        # anti-famine: si jamais vu depuis > N sec, petit boost (post)
        last = self._last_seen.get(goal_id, plan.get("created_ts", _now()))
        idle = _now() - last
        if idle > 20 * 60 and pr < 0.85:
            pr = min(1.0, pr + 0.05)
            reasons.append("anti_famine:+0.05")

        return {"priority": pr, "tags": list(tags), "explain": reasons[:6]}

    def reprioritize_all(self):
        plans: Dict[str, Dict[str, Any]] = _safe(
            self.arch, "planner", "state", "plans", default={}
        ) or {}
        if not plans:
            return
        for gid, plan in plans.items():
            if plan.get("status") == "done":
                continue
            s = self.score_goal(gid, plan)
            plan["priority"] = s["priority"]
            plan["tags"] = s["tags"]
            self._last_seen[gid] = _now()
            # trace d'explication légère (optionnel)
            try:
                self.arch.memory.add_memory(
                    {
                        "kind": "priority_trace",
                        "goal": gid,
                        "priority": round(s["priority"], 3),
                        "tags": s["tags"],
                        "why": s["explain"],
                        "ts": _now(),
                    }
                )
            except Exception:
                pass
