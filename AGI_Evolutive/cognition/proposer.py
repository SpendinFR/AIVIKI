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
