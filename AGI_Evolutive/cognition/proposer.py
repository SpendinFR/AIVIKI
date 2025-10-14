import time
from typing import Dict, Any, List

class Proposer:
    """Suggests adaptations for the self-model and policy based on drives and activity."""

    def __init__(self, memory_store, planner, homeostasis):
        self.memory = memory_store
        self.planner = planner
        self.homeo = homeostasis
        self.last_proposal_ts = 0.0

    def run_once_now(self) -> List[Dict[str, Any]]:
        drives = self.homeo.state.get("drives", {})
        proposals: List[Dict[str, Any]] = []

        if drives.get("curiosity", 0.5) < 0.4:
            proposals.append({
                "type": "adjust_drive",
                "drive": "curiosity",
                "target": min(1.0, drives.get("curiosity", 0.5) + 0.2),
                "reason": "Curiosité basse détectée"
            })
        if drives.get("competence", 0.5) < 0.4:
            proposals.append({
                "type": "adjust_drive",
                "drive": "competence",
                "target": min(1.0, drives.get("competence", 0.5) + 0.15),
                "reason": "Compétence perçue faible"
            })

        pending_goals = self.planner.pending_goals()
        if len(pending_goals) > 5:
            proposals.append({
                "type": "planning_hint",
                "action": "prioritize",
                "reason": "Beaucoup de plans en attente, suggérer une priorisation"
            })

        now = time.time()
        if proposals:
            self.last_proposal_ts = now
            self.memory.add_memory({
                "kind": "reflection",
                "text": f"{len(proposals)} propositions d'adaptation générées",
                "proposals": proposals,
                "ts": now
            })
from typing import Any, Dict, List


class Proposer:
    """Generate proposals for self-model updates based on drives and memory."""

    def __init__(self, memory_store, planner, homeostasis) -> None:
        self.memory = memory_store
        self.planner = planner
        self.homeo = homeostasis

    def run_once_now(self) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        drives = self.homeo.state["drives"]
        recent = self.memory.get_recent_memories(n=50)
        error_count = sum(1 for memo in recent if memo.get("kind") == "error")
        if drives["curiosity"] > 0.55 and error_count >= 3:
            proposals.append({"type": "update", "path": ["persona", "tone"], "value": "inquisitive-analytical"})

        if drives["social_bonding"] > 0.55:
            proposals.append(
                {
                    "type": "add",
                    "path": ["persona", "values"],
                    "value": ["growth", "truth", "help", "empathy"],
                }
            )
        return proposals
