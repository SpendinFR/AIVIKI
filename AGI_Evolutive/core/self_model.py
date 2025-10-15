import copy
import json
import os
import time
from typing import Any, Dict, List

from AGI_Evolutive.core.config import cfg
from AGI_Evolutive.utils import now_iso, safe_write_json

_DEFAULT_SELF: Dict[str, Any] = {
    "identity": {
        "name": "AGI Evolutive",
        "version": "1.0",
        "created_at": now_iso(),
    },
    "persona": {
        "tone": "helpful",
        "values": ["curiosity", "care", "precision"],
    },
    "history": [],
}


class SelfModel:
    """Persisted representation of the system identity/persona."""

    def __init__(self) -> None:
        conf = cfg()
        self.path = conf["SELF_PATH"]
        self.versions_dir = conf["SELF_VERSIONS_DIR"]
        self.state: Dict[str, Any] = copy.deepcopy(_DEFAULT_SELF)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    self.state = json.load(handle)
            except Exception:
                pass

    def _snapshot_version(self) -> None:
        os.makedirs(self.versions_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        version_path = os.path.join(self.versions_dir, f"self_model_{ts}.json")
        safe_write_json(version_path, self.state)

    def save(self) -> None:
        safe_write_json(self.path, self.state)

    def apply_proposal(self, proposal: Dict[str, Any], policy) -> Dict[str, Any]:
        decision = policy.validate_proposal(proposal, self.state)
        if decision.get("decision") != "allow":
            return decision

        path: List[str] = proposal.get("path", [])
        if not path:
            return {"decision": "deny", "reason": "path manquant"}

        target = self.state
        for key in path[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        leaf = path[-1]
        action = proposal.get("type", "update")
        value = proposal.get("value")

        self._snapshot_version()
        if action == "update":
            target[leaf] = value
        elif action == "add":
            existing = target.get(leaf)
            if isinstance(existing, list):
                if isinstance(value, list):
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                else:
                    if value not in existing:
                        existing.append(value)
            else:
                target[leaf] = value
        elif action == "remove":
            existing = target.get(leaf)
            if isinstance(existing, list) and value in existing:
                existing.remove(value)
            elif leaf in target:
                target.pop(leaf)
        else:
            target[leaf] = value

        self.state.setdefault("history", []).append(
            {
                "ts": now_iso(),
                "proposal": proposal,
                "decision": decision,
            }
        )
        self.save()
        return decision
