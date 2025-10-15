"""Light-weight structural causal model helpers and counterfactual simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .structures import CausalStore, DomainSimulator, SimulationResult


@dataclass
class CounterfactualReport:
    query: Dict[str, Any]
    supported: bool
    intervention: Dict[str, Any]
    evidence: Dict[str, Any]
    simulations: List[Dict[str, Any]]
    generated_at: float


class SCMStore(CausalStore):
    """Causal knowledge base derived from the belief graph."""

    def __init__(self, beliefs: Any, ontology: Any, *, decay: float = 0.02) -> None:
        super().__init__()
        self.beliefs = beliefs
        self.ontology = ontology
        self.decay = decay
        self._bootstrap_from_beliefs()

    # ------------------------------------------------------------------
    def _bootstrap_from_beliefs(self) -> None:
        if not self.beliefs:
            return
        try:
            for belief in self.beliefs.query(relation="causes", active_only=False):
                self.register(
                    belief.subject,
                    belief.value,
                    strength=float(belief.confidence),
                    description=belief.relation_type,
                )
        except Exception:
            # Belief graph may not expose the required API yet; fail softly.
            return

    def refresh_from_belief(self, belief: Any) -> None:
        if getattr(belief, "relation", "") != "causes":
            return
        self.register(
            belief.subject,
            belief.value,
            strength=float(getattr(belief, "confidence", 0.5)),
            description=getattr(belief, "relation_type", None),
        )


class CounterfactualSimulator:
    """Evaluate lightweight counterfactuals based on the SCM and domain sims."""

    def __init__(self, scm: SCMStore, domain: Optional[DomainSimulator] = None) -> None:
        self.scm = scm
        self.domain = domain or DomainSimulator()

    # ------------------------------------------------------------------
    def run(self, query: Dict[str, Any]) -> CounterfactualReport:
        cause = query.get("cause") or query.get("if") or query.get("action")
        effect = query.get("effect") or query.get("then")
        scenario = query.get("scenario") or {}
        evidence = self.scm.test_relation(cause, effect, context=scenario)
        intervention = self.scm.intervention(cause or "unknown", action=query.get("action"))
        simulations: List[Dict[str, Any]] = []

        domains: Iterable[str] = query.get("domains") or [cause or "generic"]
        for domain in domains:
            sim = self.domain.simulate(domain, {**scenario, "cause": cause, "effect": effect})
            if isinstance(sim, SimulationResult):
                simulations.append(sim.to_dict())
            elif isinstance(sim, dict):
                simulations.append(sim)
            else:
                simulations.append({"success": bool(sim), "outcome": str(sim)})

        return CounterfactualReport(
            query=query,
            supported=bool(evidence.get("supported")),
            intervention=intervention,
            evidence=evidence,
            simulations=simulations,
            generated_at=time.time(),
        )

