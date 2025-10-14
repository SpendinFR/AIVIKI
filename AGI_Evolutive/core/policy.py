from typing import Any, Dict


class PolicyEngine:
    """Simple policy guard for self-model proposals."""

    def validate_proposal(self, proposal: Dict[str, Any], self_state: Dict[str, Any]) -> Dict[str, Any]:
        path = proposal.get("path", [])
        if not path:
            return {"decision": "deny", "reason": "path manquant"}

        if path[0] == "core_immutable":
            return {"decision": "deny", "reason": "noyau protégé"}

        if path == ["identity", "name"] and isinstance(proposal.get("value"), str) and len(proposal["value"]) > 20:
            return {"decision": "needs_human", "reason": "changement identité important"}

        return {"decision": "allow", "reason": "OK"}
