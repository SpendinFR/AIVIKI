"""Lightweight structures representing Mechanistic Actionable Insights (MAI).

This module centralises the dataclasses shared across policy, knowledge and
runtime modules.  It provides a small ``Bid`` container used by MAIs to push
signals into the global workspace as well as helper methods to evaluate
preconditions and derive bids from stored metadata.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

Expr = Dict[str, Any]


@dataclass
class EvidenceRef:
    """Reference to supporting material for an MAI."""

    source: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    kind: Optional[str] = None
    weight: float = 1.0


@dataclass
class ImpactHypothesis:
    """Qualitative hypothesis about the impact of applying an MAI."""

    trust_delta: float = 0.0
    harm_delta: float = 0.0
    identity_coherence_delta: float = 0.0
    competence_delta: float = 0.0
    regret_delta: float = 0.0
    uncertainty: float = 0.5
    confidence: float = 0.0
    rationale: Optional[str] = None
    caveats: Optional[str] = None


@dataclass
class Bid:
    """Bid emitted by an MAI or another attentional mechanism."""

    mai_id: Optional[str]
    action_hint: str
    expected_info_gain: float = 0.0
    urgency: float = 0.0
    affect_value: float = 0.0
    cost: float = 0.0
    expires_at: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    target: Optional[Any] = None
    rationale: Optional[str] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)

    def origin_tag(self) -> str:
        origin = self.payload.get("origin")
        if origin:
            return str(origin)
        if self.mai_id:
            return f"MAI:{self.mai_id}"
        return "mechanism"

    def serialise(self) -> Dict[str, Any]:
        data = asdict(self)
        data["origin"] = self.origin_tag()
        return data


def eval_expr(expr: Expr, state: Dict[str, Any], predicate_registry: Dict[str, Callable[..., bool]]) -> bool:
    """Evaluate a minimal boolean expression tree against the provided state."""

    op = expr.get("op")
    if op == "atom":
        name = expr.get("name")
        args = expr.get("args", [])
        func = predicate_registry.get(name)
        if func is None:
            return False
        try:
            return bool(func(state, *args))
        except Exception:
            return False
    if op == "and":
        return all(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    if op == "or":
        return any(eval_expr(e, state, predicate_registry) for e in expr.get("args", []))
    if op == "not":
        args = expr.get("args", [])
        return not eval_expr(args[0], state, predicate_registry) if args else True
    return False


@dataclass
class MAI:
    """Mechanistic Actionable Insight."""

    id: str
    docstring: str = ""
    version: int = 1
    title: str = ""
    summary: str = ""
    status: str = "draft"
    expected_impact: ImpactHypothesis = field(default_factory=ImpactHypothesis)
    provenance_docs: List[EvidenceRef] = field(default_factory=list)
    provenance_episodes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    preconditions: List[Any] = field(default_factory=list)
    bids: List[Mapping[str, Any]] = field(default_factory=list)
    precondition_expr: Expr = field(default_factory=dict)
    propose_spec: Dict[str, Any] = field(default_factory=dict)
    safety_invariants: List[str] = field(default_factory=list)
    runtime_counters: Dict[str, float] = field(
        default_factory=lambda: {
            "activation": 0.0,
            "wins": 0.0,
            "benefit": 0.0,
            "regret": 0.0,
            "rollbacks": 0.0,
        }
    )

    def _iter_preconditions(self) -> Iterable[Any]:
        for cond in self.preconditions:
            yield cond
        meta_pre = self.metadata.get("preconditions") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_pre, (list, tuple)):
            for cond in meta_pre:
                yield cond

    def _iter_bid_configs(self) -> Iterable[Mapping[str, Any]]:
        if self.bids:
            for conf in self.bids:
                if isinstance(conf, Mapping):
                    yield conf
        meta_bids = self.metadata.get("bids") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_bids, list):
            for conf in meta_bids:
                if isinstance(conf, Mapping):
                    yield conf
        if not self.bids and not meta_bids:
            yield {
                "action_hint": self.metadata.get("action_hint", "ClarifyIntent"),
                "expected_info_gain": self.expected_impact.confidence,
                "affect_value": self.expected_impact.trust_delta,
                "urgency": self.metadata.get("urgency", 0.0),
                "cost": self.metadata.get("cost", 0.0),
            }
        for conf in self.propose_spec.get("bids", []):
            if isinstance(conf, Mapping):
                yield conf

    def is_applicable(self, state: Mapping[str, Any], predicate_registry: Mapping[str, Callable[..., bool]]) -> bool:
        if self.precondition_expr:
            try:
                return eval_expr(self.precondition_expr, dict(state), dict(predicate_registry))
            except Exception:
                return False
        for cond in self._iter_preconditions():
            name: Optional[str]
            args: Sequence[Any]
            negate = False
            if isinstance(cond, str):
                name = cond
                args = ()
            elif isinstance(cond, Mapping):
                name = cond.get("name") or cond.get("predicate")
                negate = bool(cond.get("negate"))
                raw_args = cond.get("args", ())
                if isinstance(raw_args, Sequence) and not isinstance(raw_args, (str, bytes)):
                    args = tuple(raw_args)
                elif raw_args is None:
                    args = ()
                else:
                    args = (raw_args,)
            else:
                continue
            pred = predicate_registry.get(name)
            if pred is None:
                return False
            try:
                result = pred(state, *args)
            except TypeError:
                result = pred(state)
            except Exception:
                result = False
            result = bool(result)
            if negate:
                result = not result
            if not result:
                return False
        return True

    def propose(self, state: Mapping[str, Any]) -> List[Bid]:
        now = time.time()
        proposals: List[Bid] = []
        for conf in self._iter_bid_configs():
            action_hint = str(conf.get("action_hint", "ClarifyIntent"))
            expected_info_gain = float(conf.get("expected_info_gain", self.expected_impact.confidence))
            urgency = float(conf.get("urgency", 0.0))
            affect_value = float(conf.get("affect_value", self.expected_impact.trust_delta))
            cost = float(conf.get("cost", 0.0))
            rationale = conf.get("rationale")
            target = conf.get("target")
            expires_at = conf.get("expires_at")
            if expires_at is None and conf.get("expires_in") is not None:
                try:
                    expires_at = now + float(conf.get("expires_in", 0.0))
                except Exception:
                    expires_at = None
            payload_source = conf.get("payload", {})
            payload = dict(payload_source) if isinstance(payload_source, Mapping) else {}
            payload.setdefault("origin", f"MAI:{self.id}")
            proposals.append(
                Bid(
                    mai_id=self.id,
                    action_hint=action_hint,
                    expected_info_gain=expected_info_gain,
                    urgency=urgency,
                    affect_value=affect_value,
                    cost=cost,
                    expires_at=expires_at,
                    payload=payload,
                    target=target,
                    rationale=rationale,
                    evidence_refs=list(self.provenance_docs),
                )
            )
        return proposals

    def update_from_feedback(self, delta: Dict[str, float]) -> None:
        for key, value in delta.items():
            self.runtime_counters[key] = self.runtime_counters.get(key, 0.0) + float(value)


def new_mai(
    docstring: str,
    precondition_expr: Expr,
    propose_spec: Dict[str, Any],
    evidence: List[EvidenceRef],
    safety: List[str],
) -> MAI:
    return MAI(
        id=str(uuid.uuid4()),
        docstring=docstring,
        precondition_expr=precondition_expr,
        propose_spec=propose_spec,
        provenance_docs=evidence,
        safety_invariants=safety,
    )


__all__ = ["EvidenceRef", "ImpactHypothesis", "Bid", "MAI", "eval_expr", "new_mai"]
