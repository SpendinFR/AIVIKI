"""Core dataclasses and helpers for policy/knowledge/runtime integration.

This module defines:
- EvidenceRef: references to supporting material.
- ImpactHypothesis: qualitative hypothesis about applying an MAI.
- Bid: lightweight container used by MAIs to propose actions to the workspace.
- MAI: Mechanistic Actionable Insight with preconditions and bid generation.

Conventions:
- 'source' fields are string tags (e.g., "MAI:<id>", "planner", "critic").
- Bid expiration uses either an absolute 'expires_at' (epoch seconds) or a
  relative duration via 'expires_in' / 'ttl_s' in configs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

# ---------- Types ----------
Expr = Dict[str, Any]  # {"op":"and","args":[...]} | {"op":"atom","name":"has_commitment","args":[...]}


# ---------- Dataclasses ----------
@dataclass
class EvidenceRef:
    """Reference to supporting material for an MAI."""
    source: Optional[str] = None            # e.g. "doc:xyz.pdf#p3", "episode:<id>"
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    kind: Optional[str] = None              # e.g. "doc", "episode", "web"
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
    source: str                              # e.g., "MAI:<id>" | "planner" | "critic"
    action_hint: str                         # e.g., "AskConsent", "ClarifyIntent", ...
    target: Optional[Any] = None
    rationale: Optional[str] = None
    expected_info_gain: float = 0.0
    urgency: float = 0.0
    affect_value: float = 0.0
    cost: float = 0.0
    expires_at: Optional[float] = None       # epoch seconds
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)

    def serialise(self) -> Dict[str, Any]:
        data = asdict(self)
        # ensure a self-describing origin is always present
        data.setdefault("payload", {})
        data["payload"].setdefault("origin", self.source)
        return data


# ---------- Expression evaluation ----------
def eval_expr(expr: Expr, state: Mapping[str, Any],
              predicate_registry: Mapping[str, Callable[..., bool]]) -> bool:
    """Evaluate a minimal boolean expression tree against the provided state.

    Supported ops:
      - {"op":"atom","name":<predicate>,"args":[...]}
      - {"op":"and","args":[...]}
      - {"op":"or","args":[...]}
      - {"op":"not","args":[<expr>]}
    Unknown ops or missing predicates evaluate to False (fail-safe).
    """
    if not isinstance(expr, Mapping):
        return False

    op = expr.get("op")
    if op == "atom":
        name = expr.get("name")
        args = expr.get("args", [])
        func = predicate_registry.get(name) if isinstance(name, str) else None
        if func is None:
            return False
        try:
            return bool(func(state, *args))
        except TypeError:
            # allow predicates with (state) signature if args provided accidentally
            try:
                return bool(func(state))
            except Exception:
                return False
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


# ---------- MAI ----------
@dataclass
class MAI:
    """Mechanistic Actionable Insight."""
    id: str
    version: int = 1
    docstring: str = ""
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

    # Preconditions (legacy list form) and/or structured expression
    preconditions: List[Any] = field(default_factory=list)
    precondition_expr: Expr = field(default_factory=dict)

    # Bid sources
    bids: List[Mapping[str, Any]] = field(default_factory=list)  # explicit inline bid configs
    propose_spec: Dict[str, Any] = field(default_factory=dict)   # {"bids":[ {...}, ... ]}

    # Safety and runtime
    safety_invariants: List[str] = field(default_factory=list)
    runtime_counters: Dict[str, float] = field(default_factory=lambda: {
        "activation": 0.0,
        "wins": 0.0,
        "benefit": 0.0,
        "regret": 0.0,
        "rollbacks": 0.0,
    })

    # ---- Helpers ----
    def _iter_preconditions(self) -> Iterable[Any]:
        yield from self.preconditions
        meta_pre = self.metadata.get("preconditions") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_pre, (list, tuple)):
            for cond in meta_pre:
                yield cond

    def _iter_bid_configs(self) -> Iterable[Mapping[str, Any]]:
        # explicit configs
        for conf in self.bids:
            if isinstance(conf, Mapping):
                yield conf

        # from metadata
        meta_bids = self.metadata.get("bids") if isinstance(self.metadata, Mapping) else None
        if isinstance(meta_bids, list):
            for conf in meta_bids:
                if isinstance(conf, Mapping):
                    yield conf

        # from propose_spec
        for conf in self.propose_spec.get("bids", []):
            if isinstance(conf, Mapping):
                yield conf

        # fallback default if nothing was provided
        if not self.bids and not meta_bids and not self.propose_spec.get("bids"):
            yield {
                "action_hint": self.metadata.get("action_hint", "ClarifyIntent"),
                "expected_info_gain": float(self.expected_impact.confidence),
                "affect_value": float(self.expected_impact.trust_delta),
                "urgency": float(self.metadata.get("urgency", 0.0)),
                "cost": float(self.metadata.get("cost", 0.0)),
            }

    # ---- Logic ----
    def is_applicable(self, state: Mapping[str, Any],
                      predicate_registry: Mapping[str, Callable[..., bool]]) -> bool:
        if self.precondition_expr:
            try:
                return eval_expr(self.precondition_expr, dict(state), dict(predicate_registry))
            except Exception:
                return False

        # legacy list of predicates
        for cond in self._iter_preconditions():
            name: Optional[str]
            args: Sequence[Any]
            negate = False

            if isinstance(cond, str):
                name, args = cond, ()
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
                # unsupported entry, treat as failing precondition
                return False

            pred = predicate_registry.get(name)
            if pred is None:
                return False
            try:
                result = pred(state, *args)
            except TypeError:
                result = pred(state)
            except Exception:
                result = False

            result = not bool(result) if negate else bool(result)
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

            # normalize expiration
            expires_at = conf.get("expires_at")
            if expires_at is None:
                rel = conf.get("expires_in", conf.get("ttl_s"))
                if rel is not None:
                    try:
                        expires_at = now + float(rel)
                    except Exception:
                        expires_at = None

            # payload with origin
            payload_raw = conf.get("payload", {})
            payload = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}
            payload.setdefault("origin", f"MAI:{self.id}")

            proposals.append(
                Bid(
                    source=f"MAI:{self.id}",
                    action_hint=action_hint,
                    target=target,
                    rationale=rationale,
                    expected_info_gain=expected_info_gain,
                    urgency=urgency,
                    affect_value=affect_value,
                    cost=cost,
                    expires_at=expires_at,
                    payload=payload,
                    evidence_refs=list(self.provenance_docs),
                )
            )
        return proposals

    def touch(self) -> None:
        self.updated_at = time.time()

    def update_from_feedback(self, delta: Mapping[str, float]) -> None:
        for key, value in delta.items():
            self.runtime_counters[key] = self.runtime_counters.get(key, 0.0) + float(value)
        self.touch()


# ---------- Factory ----------
def new_mai(docstring: str, precondition_expr: Expr, propose_spec: Dict[str, Any],
            evidence: List[EvidenceRef], safety: List[str]) -> MAI:
    return MAI(
        id=str(uuid.uuid4()),
        version=1,
        docstring=docstring,
        precondition_expr=precondition_expr,
        propose_spec=propose_spec,
        provenance_docs=evidence,
        safety_invariants=safety,
    )


__all__ = ["EvidenceRef", "ImpactHypothesis", "Bid", "MAI", "eval_expr", "new_mai"]
