"""Light-weight structural causal model helpers and counterfactual simulator."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

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

    def __init__(
        self,
        beliefs: Any,
        ontology: Any,
        *,
        decay: float = 0.02,
        min_decay: float = 0.001,
        max_decay: float = 0.25,
        max_update_step: float = 0.2,
        drift_threshold: float = 0.12,
        drift_log_size: int = 200,
    ) -> None:
        super().__init__()
        self.beliefs = beliefs
        self.ontology = ontology
        self.base_decay = decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.max_update_step = max(0.01, max_update_step)
        self.decay_learning_rate = 0.08
        self.interval_smoothing = 0.25
        self.hazard_blend = 0.35
        self.drift_threshold = drift_threshold
        self.support_threshold = 0.05
        self._link_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._drift_log: Deque[Dict[str, Any]] = deque(maxlen=drift_log_size)
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

    # ------------------------------------------------------------------
    def register(
        self,
        cause: str,
        effect: str,
        *,
        strength: float = 0.5,
        description: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Register or update a causal relation with adaptive metadata."""

        key = (cause, effect)
        now = time.time()
        metadata = self._link_metadata.get(key)

        link = None
        for candidate in self._by_cause.get(cause, []):
            if candidate.effect == effect:
                link = candidate
                break

        if metadata is None or link is None:
            link = link or self._create_link(
                cause,
                effect,
                strength=float(strength),
                description=description,
                conditions=conditions,
            )
            self._link_metadata[key] = {
                "link": link,
                "strength": self._clamp01(float(strength)),
                "decay": float(self.base_decay),
                "last_update": now,
                "last_observation": now,
                "avg_interval": None,
                "usage_count": 1,
                "drift": 0.0,
                "confidence": float(confidence) if confidence is not None else float(strength),
            }
            return

        self._apply_time_decay(key, now)

        previous_strength = metadata["strength"]
        incoming_strength = float(strength)
        error = incoming_strength - previous_strength
        step = self._bounded_step(error)
        updated_strength = self._clamp01(previous_strength + step)
        metadata["strength"] = updated_strength
        metadata["confidence"] = (
            float(confidence)
            if confidence is not None
            else 0.7 * metadata.get("confidence", updated_strength) + 0.3 * incoming_strength
        )
        metadata["last_update"] = now

        previous_observation = metadata.get("last_observation", now)
        interval = now - previous_observation
        if interval > 0:
            avg_interval = metadata.get("avg_interval")
            if avg_interval is None:
                metadata["avg_interval"] = interval
            else:
                metadata["avg_interval"] = (1 - self.interval_smoothing) * avg_interval + self.interval_smoothing * interval
        metadata["last_observation"] = now
        metadata["usage_count"] = metadata.get("usage_count", 0) + 1

        self._update_decay(metadata, reinforcement=step)
        self._record_drift(key, updated_strength - previous_strength, reason="reinforcement")

        link.strength = metadata["strength"]
        if description:
            link.description = description
        if conditions is not None:
            link.conditions = list(conditions)

    # ------------------------------------------------------------------
    def _create_link(
        self,
        cause: str,
        effect: str,
        *,
        strength: float,
        description: Optional[str],
        conditions: Optional[List[str]],
    ):
        from .structures import CausalLink

        link = CausalLink(
            cause=cause,
            effect=effect,
            strength=strength,
            description=description,
            conditions=list(conditions) if conditions else None,
        )
        self.add_link(link)
        return link

    def _bounded_step(self, delta: float) -> float:
        if delta > self.max_update_step:
            return self.max_update_step
        if delta < -self.max_update_step:
            return -self.max_update_step
        return delta

    def _clamp01(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _update_decay(self, metadata: Dict[str, Any], *, reinforcement: float) -> None:
        current = metadata.get("decay", self.base_decay)
        adjusted = current - self.decay_learning_rate * reinforcement
        adjusted = max(self.min_decay, min(self.max_decay, adjusted))

        avg_interval = metadata.get("avg_interval")
        if avg_interval:
            hazard = 1.0 / max(avg_interval, 1e-6)
            adjusted = (1 - self.hazard_blend) * adjusted + self.hazard_blend * hazard
        metadata["decay"] = max(self.min_decay, min(self.max_decay, adjusted))

    def _apply_time_decay(self, key: Tuple[str, str], now: Optional[float] = None) -> None:
        metadata = self._link_metadata.get(key)
        if not metadata:
            return
        now = now or time.time()
        last_update = metadata.get("last_update", now)
        elapsed = max(0.0, now - last_update)
        if elapsed <= 0.0:
            return
        decay = metadata.get("decay", self.base_decay)
        decayed_strength = metadata["strength"] * math.exp(-decay * elapsed)
        decayed_strength = self._clamp01(decayed_strength)
        if abs(decayed_strength - metadata["strength"]) < 1e-6:
            metadata["last_update"] = now
            return

        previous_strength = metadata["strength"]
        metadata["strength"] = decayed_strength
        metadata["last_update"] = now
        metadata["link"].strength = decayed_strength
        self._record_drift(key, decayed_strength - previous_strength, reason="decay")

    def _record_drift(self, key: Tuple[str, str], delta: float, *, reason: str) -> None:
        metadata = self._link_metadata.get(key)
        if not metadata:
            return
        drift = metadata.get("drift", 0.0)
        metadata["drift"] = 0.8 * drift + 0.2 * abs(delta)
        magnitude = abs(delta)
        if magnitude < self.drift_threshold:
            return
        cause, effect = key
        self._drift_log.append(
            {
                "cause": cause,
                "effect": effect,
                "delta": delta,
                "reason": reason,
                "timestamp": time.time(),
                "strength": metadata.get("strength"),
                "decay": metadata.get("decay"),
            }
        )

    # ------------------------------------------------------------------
    def test_relation(
        self,
        cause: str,
        effect: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        now = time.time()
        links = [link for link in self.get_effects(cause) if link.effect == effect]
        enriched: List[Dict[str, Any]] = []
        supported = False
        for link in links:
            key = (link.cause, link.effect)
            self._apply_time_decay(key, now)
            payload = link.to_dict()
            metadata = self._link_metadata.get(key)
            if metadata:
                payload.update(
                    {
                        "decay": metadata.get("decay"),
                        "usage_count": metadata.get("usage_count"),
                        "drift": metadata.get("drift"),
                        "last_update": metadata.get("last_update"),
                        "avg_interval": metadata.get("avg_interval"),
                    }
                )
                supported = supported or metadata["strength"] >= self.support_threshold
            enriched.append(payload)

        satisfied: List[str] = []
        unsatisfied: List[str] = []
        for link in links:
            for cond in link.conditions or []:
                key, _, expected = cond.partition("=")
                key = key.strip()
                expected = expected.strip()
                value = str(context.get(key, ""))
                if expected:
                    (satisfied if value == expected else unsatisfied).append(cond)
                elif value:
                    satisfied.append(cond)
                else:
                    unsatisfied.append(cond)

        return {
            "cause": cause,
            "effect": effect,
            "supported": supported,
            "links": enriched,
            "satisfied_conditions": satisfied,
            "unsatisfied_conditions": unsatisfied,
        }

    # ------------------------------------------------------------------
    def intervention(self, cause: str, action: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        effects = self.get_effects(cause)
        predicted = []
        for link in effects:
            key = (link.cause, link.effect)
            self._apply_time_decay(key, now)
            metadata = self._link_metadata.get(key, {})
            predicted.append(
                {
                    "effect": link.effect,
                    "strength": float(link.strength),
                    "conditions": list(link.conditions or []),
                    "decay": metadata.get("decay"),
                    "drift": metadata.get("drift"),
                }
            )
        return {
            "cause": cause,
            "action": action or f"do({cause})",
            "predicted_effects": predicted,
        }

    # ------------------------------------------------------------------
    def ingest_simulation(
        self, cause: Optional[str], effect: Optional[str], result: Union[SimulationResult, Dict[str, Any], None]
    ) -> None:
        if not cause or not effect or result is None:
            return
        key = (cause, effect)
        if key not in self._link_metadata:
            return

        now = time.time()
        self._apply_time_decay(key, now)
        metadata = self._link_metadata[key]

        success = False
        confidence = 1.0
        if isinstance(result, SimulationResult):
            success = bool(result.success)
            if result.details and "confidence" in result.details:
                try:
                    confidence = float(result.details["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
            if result.details and "likelihood" in result.details:
                try:
                    confidence = float(result.details["likelihood"])
                except (TypeError, ValueError):
                    pass
        elif isinstance(result, dict):
            success = bool(result.get("success"))
            if "confidence" in result:
                try:
                    confidence = float(result["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
        else:
            success = bool(result)

        direction = 1.0 if success else -1.0
        feedback_scale = 0.05
        step = self._bounded_step(direction * confidence * feedback_scale)
        previous_strength = metadata["strength"]
        metadata["strength"] = self._clamp01(previous_strength + step)
        metadata["last_update"] = now
        metadata["usage_count"] = metadata.get("usage_count", 0) + 1
        metadata["link"].strength = metadata["strength"]

        previous_observation = metadata.get("last_observation", now)
        interval = now - previous_observation
        if interval > 0:
            avg_interval = metadata.get("avg_interval")
            if avg_interval is None:
                metadata["avg_interval"] = interval
            else:
                metadata["avg_interval"] = (1 - self.interval_smoothing) * avg_interval + self.interval_smoothing * interval
        metadata["last_observation"] = now

        self._update_decay(metadata, reinforcement=step)
        self._record_drift(key, metadata["strength"] - previous_strength, reason="simulation")

    # ------------------------------------------------------------------
    @property
    def drift_log(self) -> List[Dict[str, Any]]:
        return list(self._drift_log)

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
            self.scm.ingest_simulation(cause, effect, sim)
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

