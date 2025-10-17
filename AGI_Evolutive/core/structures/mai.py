from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import uuid

# --------- Evidence / Impact / Bid ---------
@dataclass
class EvidenceRef:
    source: str                 # "doc:xyz.pdf#p32:offset" | "episode:<id>"
    snippet: Optional[str] = None
    weight: float = 1.0

@dataclass
class ImpactHypothesis:
    # Direction attendue sur quelques grandeurs générales (non exhaustives, extensibles)
    trust_delta: float = 0.0
    harm_delta: float = 0.0
    identity_coherence_delta: float = 0.0
    competence_delta: float = 0.0
    regret_delta: float = 0.0
    uncertainty: float = 0.5        # incertitude sur l’hypothèse

@dataclass
class Bid:
    source: str                  # "MAI:<id>" | "planner" | "critic" | ...
    action_hint: str            # ex: "AskConsent", "PreferPlan:GatherEvidenceFirst", "PartialReveal", "Don’tActYet"
    target: Optional[Any] = None
    rationale: Optional[str] = None
    expected_info_gain: float = 0.0
    urgency: float = 0.0
    affect_value: float = 0.0
    cost: float = 0.0
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    expires_at: Optional[float] = None

# --------- Expression (préconditions) ---------
# Mini DSL booléenne (AND/OR/NOT/ATOM) sérialisable → evaluation sur "state"
Expr = Dict[str, Any]  # {"op":"and","args":[...]} | {"op":"atom","name":"has_commitment","args":[...]}

def eval_expr(expr: Expr, state: Dict[str, Any], predicate_registry: Dict[str, Callable[..., bool]]) -> bool:
    op = expr.get("op")
    if op == "atom":
        name = expr["name"]
        args = expr.get("args", [])
        func = predicate_registry.get(name)
        if func is None:
            return False  # inconnu => par défaut faux (sûr)
        try:
            return bool(func(state, *args))
        except Exception:
            return False
    elif op == "and":
        return all(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    elif op == "or":
        return any(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    elif op == "not":
        args = expr.get("args", [])
        return not eval_expr(args[0], state, predicate_registry) if args else True
    return False

# --------- MAI ---------
@dataclass
class MAI:
    id: str
    version: int
    docstring: str
    provenance_docs: List[EvidenceRef] = field(default_factory=list)
    provenance_episodes: List[str] = field(default_factory=list)

    # Précondition: expression JSON + registre de prédicats (injecté par l’intégration)
    precondition_expr: Expr = field(default_factory=dict)

    # Propositions: une fonction “générique” encapsulée par un spec JSON + builder
    propose_spec: Dict[str, Any] = field(default_factory=dict)

    # Hypothèse d’impact (parsée depuis l’induction, ajustée en ligne)
    expected_impact: ImpactHypothesis = field(default_factory=ImpactHypothesis)

    # Invariants de sécurité (ex: "keep_promise", "no_personal_data_without_consent")
    safety_invariants: List[str] = field(default_factory=list)

    # Statistiques vivantes (créées/retirées dynamiquement)
    runtime_counters: Dict[str, float] = field(default_factory=lambda: {
        "activation": 0.0, "wins": 0.0, "benefit": 0.0, "regret": 0.0, "rollbacks": 0.0
    })

    # ---- Exécution ----
    def is_applicable(self, state: Dict[str, Any], predicates: Dict[str, Callable[..., bool]]) -> bool:
        if not self.precondition_expr:
            return False
        return eval_expr(self.precondition_expr, state, predicates)

    def propose(self, state: Dict[str, Any]) -> List[Bid]:
        """
        propose_spec: {
          "bids": [
            {"action_hint":"AskConsent","rationale":"…"},
            {"action_hint":"PartialReveal","target":{"redact": ["pii","secret"]}}
          ]
        }
        """
        bids = []
        for b in self.propose_spec.get("bids", []):
            bids.append(Bid(
                source=f"MAI:{self.id}",
                action_hint=b.get("action_hint",""),
                target=b.get("target"),
                rationale=b.get("rationale"),
                expected_info_gain=float(b.get("expected_info_gain", 0.0)),
                urgency=float(b.get("urgency", 0.0)),
                affect_value=float(b.get("affect_value", 0.0)),
                cost=float(b.get("cost", 0.0)),
                evidence_refs=self.provenance_docs,
                expires_at=time.time() + float(b.get("ttl_s", 5.0))
            ))
        return bids

    def update_from_feedback(self, delta: Dict[str, float]) -> None:
        # Ajuste les compteurs; l’inducer/critic peut aussi remplacer precondition/propose_spec si besoin
        for k, v in delta.items():
            self.runtime_counters[k] = self.runtime_counters.get(k, 0.0) + float(v)

# --------- Utilitaires ---------
def new_mai(docstring: str, precondition_expr: Expr, propose_spec: Dict[str, Any],
            evidence: List[EvidenceRef], safety: List[str]) -> MAI:
    return MAI(
        id=str(uuid.uuid4()),
        version=1,
        docstring=docstring,
        provenance_docs=evidence,
        precondition_expr=precondition_expr,
        propose_spec=propose_spec,
        safety_invariants=safety,
    )
