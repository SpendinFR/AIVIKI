from typing import Optional, Dict, Any, List
import time
import random
from .dag_store import DagStore, GoalNode
from .curiosity import CuriosityEngine


class GoalSystem:
    """
    Gestionnaire d'objectifs à DAG + curiosité (info-gain + ZPD).
    Intégration: memory / reasoning / metacognition / emotions / reward_engine.
    """

    def __init__(self, architecture=None, memory=None, reasoning=None):
        self.arch = architecture
        self.memory = memory
        self.reasoning = reasoning
        self.store = DagStore(persist_path="data/goals.json", dashboard_path="data/goals_dashboard.json")
        self.curiosity = CuriosityEngine(architecture=self.arch)
        self.active_goal_id: Optional[str] = self.store.active_goal_id
        self.last_auto_proposal_at = 0.0
        self.auto_proposal_interval = 180.0  # toutes les 3 minutes par défaut

        if len(self.store.nodes) == 0:
            root = self.store.add_goal(
                description="Évoluer (comprendre, apprendre, s’améliorer).",
                criteria=["Montrer une amélioration stable sur ≥ 2 métriques clés."],
                created_by="system",
                value=0.8,
                competence=0.5,
                curiosity=0.7,
                urgency=0.4,
            )
            self.set_active_goal(root.id)

    # ---------- API publique (appelée ailleurs) ----------
    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        node = self.store.get_active()
        return self._node_to_dict(node) if node else None

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        node = self.store.update_goal(goal_id, updates)
        return self._node_to_dict(node) if node else None

    def add_goal(self, description: str, **kwargs) -> Dict[str, Any]:
        node = self.store.add_goal(description=description, **kwargs)
        return self._node_to_dict(node)

    def set_active_goal(self, goal_id: Optional[str]) -> Optional[Dict[str, Any]]:
        node = self.store.set_active(goal_id)
        self.active_goal_id = self.store.active_goal_id
        return self._node_to_dict(node) if node else None

    def complete_goal(self, goal_id: str, success: bool = True, note: str = ""):
        self.store.complete_goal(goal_id, success=success, note=note)
        if self.memory and hasattr(self.memory, "add_memory"):
            self.memory.add_memory(
                "goal_event",
                {"t": time.time(), "type": "complete", "goal_id": goal_id, "success": success, "note": note},
            )
        m = getattr(self.arch, "metacognition", None)
        if m and hasattr(m, "_record_metacognitive_event"):
            m._record_metacognitive_event(
                event_type="goal_completed",
                domain=getattr(m, "CognitiveDomain", None).LEARNING if hasattr(m, "CognitiveDomain") else None,
                description=f"Goal {goal_id} completed: success={success}",
                significance=0.7 if success else 0.4,
                confidence=0.7,
            )

    def propose_goals(self, k: int = 3) -> List[Dict[str, Any]]:
        active = self.store.get_active()
        props = self.curiosity.suggest_subgoals(self._node_to_dict(active) if active else None, k=k)
        return props

    def get_status(self) -> Dict[str, Any]:
        act = self.store.get_active()
        return {
            "active_goal": self._node_to_dict(act) if act else None,
            "top5": [self._node_to_dict(n) for n in self.store.topk(5, only_pending=False)],
        }

    # ---------- Boucle principale (à appeler à chaque cycle) ----------
    def step(self, user_msg: Optional[str] = None):
        """
        - Recalcule priorités (value/urgency/curiosity/competence shaping)
        - S'il n'y a pas d'objectif actif: choisit le meilleur pending.
        - Toutes les X secondes: propose des sous-buts (curiosity) reliés à l'actif.
        - Ajuste légèrement progress/competence selon feedback récents (si dispo).
        """

        self.store.recompute_all_priorities()

        if not self.store.get_active():
            top = self.store.topk(1, only_pending=True)
            if top:
                self.set_active_goal(top[0].id)

        now = time.time()
        if now - self.last_auto_proposal_at > self.auto_proposal_interval:
            self.last_auto_proposal_at = now
            active = self.store.get_active()
            proposals = self.curiosity.suggest_subgoals(self._node_to_dict(active) if active else None, k=3)
            created_ids = []
            for p in proposals:
                node = self.store.add_goal(**p)
                if active:
                    self.store.link(active.id, node.id)
                created_ids.append(node.id)
                self._log_goal_creation(node, reason="curiosity_auto")
            if not self.store.get_active():
                children = [self.store.get_goal(gid) for gid in created_ids]
                children = [c for c in children if c]
                children.sort(key=lambda n: n.priority, reverse=True)
                if children:
                    self.set_active_goal(children[0].id)

        try:
            logpath = "data/logs/social_feedback.jsonl"
            delta_prog, delta_comp = 0.0, 0.0
            if os.path.exists(logpath):
                with open(logpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-5:]
                for line in lines:
                    import json

                    ev = json.loads(line)
                    r = float(ev.get("extrinsic_reward", 0.0))
                    delta_prog += 0.02 * r
                    delta_comp += 0.01 * r
            act = self.store.get_active()
            if act and (abs(delta_prog) > 1e-6 or abs(delta_comp) > 1e-6):
                self.store.update_goal(
                    act.id,
                    {
                        "progress": float(max(0.0, min(1.0, act.progress + delta_prog))),
                        "competence": float(max(0.0, min(1.0, act.competence + delta_comp))),
                    },
                )
        except Exception:
            pass

    # ---------- Internals ----------
    def _node_to_dict(self, n: GoalNode) -> Dict[str, Any]:
        if not n:
            return {}
        return {
            "id": n.id,
            "description": n.description,
            "criteria": n.criteria,
            "progress": n.progress,
            "value": n.value,
            "competence": n.competence,
            "curiosity": n.curiosity,
            "urgency": n.urgency,
            "priority": n.priority,
            "status": n.status,
            "created_by": n.created_by,
            "parent_ids": list(n.parent_ids),
            "child_ids": list(n.child_ids),
            "updated_at": n.updated_at,
        }

    def _log_goal_creation(self, node: GoalNode, reason: str):
        if self.memory and hasattr(self.memory, "add_memory"):
            self.memory.add_memory(
                "goal_event",
                {
                    "t": time.time(),
                    "type": "create",
                    "goal_id": node.id,
                    "description": node.description,
                    "reason": reason,
                },
            )
        m = getattr(self.arch, "metacognition", None)
        if m and hasattr(m, "_record_metacognitive_event"):
            m._record_metacognitive_event(
                event_type="goal_created",
                domain=getattr(m, "CognitiveDomain", None).LEARNING if hasattr(m, "CognitiveDomain") else None,
                description=f"New goal: {node.description}",
                significance=min(0.6 + 0.3 * node.curiosity, 1.0),
                confidence=0.7,
            )


import os
