from __future__ import annotations

import json
import os
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from AGI_Evolutive.utils.jsonsafe import json_sanitize


def _now() -> float:
    return time.time()


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(json_sanitize(value), ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _unique_keywords(text: str, min_len: int = 4) -> List[str]:
    tokens = []
    for chunk in text.replace("_", " ").split():
        clean = "".join(ch for ch in chunk if ch.isalnum()).lower()
        if len(clean) >= min_len:
            tokens.append(clean)
    seen: Dict[str, bool] = {}
    unique: List[str] = []
    for token in tokens:
        if token not in seen:
            seen[token] = True
            unique.append(token)
    return unique


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {
            "true",
            "yes",
            "y",
            "ok",
            "success",
            "succès",
            "réussi",
            "succeeded",
            "valide",
            "validé",
            "accepte",
            "accepté",
        }:
            return True
        if lowered in {"false", "no", "n", "fail", "failed", "échec", "ko", "invalid", "rejeté", "reject", "refus"}:
            return False
    return None


def _first_text(data: Any, *candidates: str) -> Optional[str]:
    if not data:
        return None
    source_dict = data if isinstance(data, dict) else None
    if source_dict is None and hasattr(data, "__dict"):
        try:
            source_dict = dict(data.__dict__)
        except Exception:
            source_dict = None
    for key in candidates:
        if source_dict and key in source_dict:
            value = source_dict.get(key)
        else:
            value = getattr(data, key, None)
        if value:
            return _normalise_text(value)
    return None


@dataclass
class SkillTrial:
    index: int
    coverage: float
    success: bool
    evidence: List[str] = field(default_factory=list)
    mode: str = "coverage"
    summary: Optional[str] = None
    feedback: Optional[str] = None


@dataclass
class SkillRequest:
    identifier: str
    action_type: str
    description: str
    payload: Dict[str, Any]
    created_at: float
    status: str = "pending"
    attempts: int = 0
    successes: int = 0
    trials: List[SkillTrial] = field(default_factory=list)
    knowledge: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    approval_required: bool = True
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None
    last_error: Optional[str] = None

    def public_view(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "action_type": self.action_type,
            "description": self.description,
            "status": self.status,
            "attempts": self.attempts,
            "successes": self.successes,
            "requirements": list(self.requirements),
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "approval_notes": self.approval_notes,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "knowledge_items": len(self.knowledge),
        }


class SkillSandboxManager:
    """Coordinate autonomous acquisition of new action handlers."""

    def __init__(
        self,
        storage_dir: str = "data/skills",
        *,
        min_trials: int = 3,
        success_threshold: float = 0.75,
        max_attempts: int = 5,
        approval_required: bool = True,
        run_async: bool = True,
        training_interval: float = 1.0,
    ) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "skills_index.json")
        self.min_trials = max(1, int(min_trials))
        self.success_threshold = max(0.0, min(1.0, float(success_threshold)))
        self.max_attempts = max(1, int(max_attempts))
        self.approval_required_default = bool(approval_required)
        self.run_async = bool(run_async)
        self.training_interval = max(0.0, float(training_interval))

        self._lock = threading.Lock()
        self._requests: Dict[str, SkillRequest] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._active_handlers: Dict[str, Any] = {}

        self._positive_markers: Tuple[str, ...] = (
            "succès",
            "réussi",
            "réussie",
            "réussite",
            "validé",
            "valide",
            "maîtrisé",
            "maitrise",
            "maîtrise",
            "ready",
            "prêt",
            "complet",
            "acquis",
            "ok",
            "passé",
            "passed",
        )
        self._negative_markers: Tuple[str, ...] = (
            "échec",
            "echec",
            "failed",
            "fail",
            "insuffisant",
            "insufficient",
            "pas prêt",
            "pas pret",
            "not ready",
            "refus",
            "rejet",
            "reject",
            "incomplet",
            "impossible",
            "bloqué",
            "blocked",
        )

        self.memory: Optional[Any] = None
        self.language: Optional[Any] = None
        self.simulator: Optional[Any] = None
        self.jobs: Optional[Any] = None
        self.arch_ref: Optional[weakref.ReferenceType] = None
        self.interface_ref: Optional[weakref.ReferenceType] = None

        self._load_state()

    # ------------------------------------------------------------------
    # Binding helpers
    def bind(
        self,
        *,
        memory: Optional[Any] = None,
        language: Optional[Any] = None,
        simulator: Optional[Any] = None,
        jobs: Optional[Any] = None,
        arch: Optional[Any] = None,
        interface: Optional[Any] = None,
    ) -> None:
        if memory is not None:
            self.memory = memory
        if language is not None:
            self.language = language
        if simulator is not None:
            self.simulator = simulator
        if jobs is not None:
            self.jobs = jobs
        if arch is not None:
            self.arch_ref = weakref.ref(arch)
        if interface is not None:
            self.interface_ref = weakref.ref(interface)

    # ------------------------------------------------------------------
    def register_intention(
        self,
        *,
        action_type: str,
        description: str,
        payload: Optional[Dict[str, Any]] = None,
        requirements: Optional[Sequence[str]] = None,
        knowledge: Optional[Sequence[Mapping[str, Any]]] = None,
        approval_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Expose an explicit way to register a new autonomous skill intention.

        This mirrors :meth:`_ensure_request` but allows other modules (meta-
        cognition, evolution manager, etc.) to declaratively seed new skills
        without going through the ActionInterface first.  The method is fully
        idempotent: calling it repeatedly with the same ``action_type`` updates
        the stored request with merged requirements/payload.
        """

        if not action_type:
            raise ValueError("action_type must be provided")
        action_key = str(action_type).strip()
        if not action_key:
            raise ValueError("action_type must not be empty")

        description_text = description.strip() if description else action_key.replace("_", " ")
        payload_map = dict(payload or {})

        with self._lock:
            request = self._requests.get(action_key)
            if request is None:
                request = SkillRequest(
                    identifier=str(uuid.uuid4()),
                    action_type=action_key,
                    description=description_text,
                    payload=dict(payload_map),
                    created_at=_now(),
                    status="pending",
                    approval_required=self.approval_required_default
                    if approval_required is None
                    else bool(approval_required),
                )
                request.requirements = self._extract_requirements(request)
                self._requests[action_key] = request
                self._memorise_event(
                    "skill_intention_registered",
                    {
                        "action_type": action_key,
                        "description": description_text,
                        "requirements": list(request.requirements),
                    },
                )
            else:
                if description_text and description_text != request.description:
                    request.description = description_text
                if approval_required is not None:
                    request.approval_required = bool(approval_required)
                if payload_map:
                    try:
                        request.payload.update(payload_map)
                    except Exception:
                        request.payload = dict(payload_map)

            if requirements:
                merged = list(dict.fromkeys(list(request.requirements) + [str(r) for r in requirements if r]))
                request.requirements = merged[:20]
            else:
                request.requirements = self._extract_requirements(request)

            if knowledge:
                existing = list(request.knowledge)
                for item in knowledge:
                    try:
                        existing.append(dict(item))
                    except Exception:
                        continue
                request.knowledge = existing[:20]

            self._save_state_locked()

        # Trigger training asynchronously if needed
        if request.status in {"pending", "failed"}:
            self._ensure_training(action_key)

        return request.public_view()

    # ------------------------------------------------------------------
    # Public API used by ActionInterface
    def build_handler(self, act: Any, interface: Any) -> Optional[Any]:
        if interface is not None:
            self.bind(interface=interface)

        request = self._ensure_request(act)
        if request is None:
            return None

        if request.status == "active":
            handler = self._active_handlers.get(request.action_type)
            if handler is None:
                handler = self._build_live_handler(request.action_type)
                if handler is not None:
                    self._active_handlers[request.action_type] = handler
            return handler

        if request.status in {"pending", "training"}:
            self._ensure_training(request.action_type)
            return lambda _: {
                "ok": False,
                "reason": "skill_training_in_progress",
                "skill": request.public_view(),
            }

        if request.status == "awaiting_approval":
            return lambda _: {
                "ok": False,
                "reason": "skill_waiting_user_approval",
                "skill": request.public_view(),
            }

        if request.status == "rejected":
            return lambda _: {
                "ok": False,
                "reason": "skill_rejected",
                "skill": request.public_view(),
            }

        if request.status == "failed":
            self._ensure_training(request.action_type)
            return lambda _: {
                "ok": False,
                "reason": "skill_training_retry",
                "skill": request.public_view(),
            }

        return None

    def handle_simulation(self, act: Any, interface: Any) -> Optional[Dict[str, Any]]:
        if interface is not None:
            self.bind(interface=interface)

        request = self._ensure_request(act)
        if request is None:
            return None

        if request.status == "active":
            try:
                payload = dict(getattr(act, "payload", {}) or {})
            except Exception:
                payload = {}
            return self.execute(request.action_type, payload)

        if request.status in {"pending", "training", "failed"}:
            self._ensure_training(request.action_type)
            status = self.status(request.action_type)
            reason = (
                "skill_training_retry" if request.status == "failed" else "skill_training_in_progress"
            )
            return {"ok": False, "reason": reason, "skill": status}

        if request.status == "awaiting_approval":
            return {
                "ok": False,
                "reason": "skill_waiting_user_approval",
                "skill": self.status(request.action_type),
            }

        if request.status == "rejected":
            return {
                "ok": False,
                "reason": "skill_rejected",
                "skill": self.status(request.action_type),
            }

        return None

    def review(
        self,
        action_type: str,
        decision: str,
        reviewer: Optional[str] = None,
        notes: Optional[str] = None,
        *,
        interface: Optional[Any] = None,
    ) -> Dict[str, Any]:
        decision = (decision or "").strip().lower()
        if not action_type:
            return {"ok": False, "reason": "missing_action_type"}

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"ok": False, "reason": "unknown_skill"}
            if request.status != "awaiting_approval":
                return {"ok": False, "reason": "skill_not_ready", "skill": request.public_view()}

            if decision not in {"approve", "approved", "accept", "reject", "rejected", "deny"}:
                return {"ok": False, "reason": "invalid_decision"}

            if decision in {"reject", "rejected", "deny"}:
                request.status = "rejected"
                request.approved_by = reviewer
                request.approval_notes = notes
                self._save_state_locked()
                return {"ok": True, "status": "rejected", "skill": request.public_view()}

            request.status = "active"
            request.approved_by = reviewer
            request.approval_notes = notes
            self._save_state_locked()

        handler = self._build_live_handler(action_type, interface=interface)
        if handler is not None:
            self._active_handlers[action_type] = handler
            bound_interface = interface
            if bound_interface is None and self.interface_ref is not None:
                bound_interface = self.interface_ref()
            if bound_interface is not None and hasattr(bound_interface, "register_handler"):
                bound_interface.register_handler(action_type, handler)

        self._memorise_event(
            "skill_approved",
            {
                "action_type": action_type,
                "reviewer": reviewer,
                "notes": notes,
            },
        )

        return {"ok": True, "status": "active", "skill": self.status(action_type)}

    def status(self, action_type: str) -> Dict[str, Any]:
        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"action_type": action_type, "status": "missing"}
            payload = request.public_view()
            trials = [
                {
                    "index": t.index,
                    "coverage": t.coverage,
                    "success": t.success,
                    "mode": t.mode,
                    "summary": t.summary,
                    "feedback": t.feedback,
                    "evidence": list(t.evidence),
                }
                for t in request.trials
            ]
            payload.update(
                {
                    "trials": trials,
                    "success_rate": self._success_rate(request),
                }
            )
            return payload

    def list_skills(
        self,
        status: Optional[str] = None,
        *,
        include_trials: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return a snapshot of tracked skills optionally filtered by status."""

        with self._lock:
            requests = list(self._requests.values())

        requests.sort(key=lambda req: req.created_at)

        snapshot: List[Dict[str, Any]] = []
        for request in requests:
            if status and request.status != status:
                continue

            payload = request.public_view()
            payload["success_rate"] = self._success_rate(request)
            if include_trials:
                payload["trials"] = [
                    {
                        "index": t.index,
                        "coverage": t.coverage,
                        "success": t.success,
                        "mode": t.mode,
                        "summary": t.summary,
                        "feedback": t.feedback,
                        "evidence": list(t.evidence),
                    }
                    for t in request.trials
                ]
            else:
                payload["trial_count"] = len(request.trials)

            snapshot.append(payload)

        return snapshot

    # ------------------------------------------------------------------
    # Execution
    def execute(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return {"ok": False, "reason": "unknown_skill"}
            if request.status != "active":
                return {
                    "ok": False,
                    "reason": "skill_not_active",
                    "skill": request.public_view(),
                }
            knowledge = list(request.knowledge)

        summary = self._render_execution_summary(request, payload, knowledge)
        return {
            "ok": True,
            "skill": action_type,
            "summary": summary,
            "knowledge_used": knowledge,
            "trials": [
                {
                    "index": t.index,
                    "coverage": t.coverage,
                    "success": t.success,
                    "mode": t.mode,
                    "summary": t.summary,
                    "feedback": t.feedback,
                    "evidence": list(t.evidence),
                }
                for t in request.trials
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_request(self, act: Any) -> Optional[SkillRequest]:
        action_type = getattr(act, "type", None)
        payload = getattr(act, "payload", {}) or {}
        context = getattr(act, "context", {}) or {}
        if not action_type:
            return None

        with self._lock:
            if action_type in self._requests:
                request = self._requests[action_type]
            else:
                description = _normalise_text(
                    payload.get("description")
                    or context.get("description")
                    or payload.get("goal")
                    or context.get("goal")
                    or action_type.replace("_", " ")
                )
                request = SkillRequest(
                    identifier=str(uuid.uuid4()),
                    action_type=action_type,
                    description=description,
                    payload=dict(payload),
                    created_at=_now(),
                    status="pending",
                    approval_required=self.approval_required_default,
                )
                request.requirements = self._extract_requirements(request)
                self._requests[action_type] = request
                self._save_state_locked()
                self._memorise_event(
                    "skill_requested",
                    {
                        "action_type": action_type,
                        "description": description,
                        "requirements": request.requirements,
                    },
                )
            return request

    def _ensure_training(self, action_type: str) -> None:
        if not self.run_async:
            self._training_loop(action_type)
            return

        with self._lock:
            thread = self._threads.get(action_type)
            if thread and thread.is_alive():
                return

            thread = threading.Thread(
                target=self._training_loop,
                args=(action_type,),
                name=f"skill-train-{action_type}",
                daemon=True,
            )
            self._threads[action_type] = thread
            thread.start()

    def _training_loop(self, action_type: str) -> None:
        for _ in range(self.max_attempts):
            with self._lock:
                request = self._requests.get(action_type)
                if request is None:
                    return
                if request.status == "active":
                    return
                request.status = "training"
                request.attempts += 1
                self._save_state_locked()

            knowledge = self._gather_knowledge(request)
            trials, successes = self._run_trials(request, knowledge)

            with self._lock:
                request = self._requests.get(action_type)
                if request is None:
                    return
                request.knowledge = knowledge
                request.trials = trials
                request.successes = successes
                success_rate = self._success_rate(request)
                if success_rate >= self.success_threshold:
                    request.status = "awaiting_approval" if request.approval_required else "active"
                    self._save_state_locked()
                    self._notify_ready(request)
                    return
                request.status = "failed"
                request.last_error = "insufficient_success_rate"
                self._save_state_locked()

            if self.training_interval > 0:
                time.sleep(self.training_interval)

        with self._lock:
            request = self._requests.get(action_type)
            if request is None:
                return
            if request.status not in {"awaiting_approval", "active"}:
                request.status = "failed"
                request.last_error = "max_attempts_reached"
                self._save_state_locked()

    def _gather_knowledge(self, request: SkillRequest) -> List[Dict[str, Any]]:
        query = request.description or request.action_type.replace("_", " ")
        knowledge: List[Dict[str, Any]] = []
        memory = self.memory
        if memory is not None:
            try:
                if hasattr(memory, "search"):
                    hits = memory.search(query, top_k=8)
                elif hasattr(memory, "get_recent_memories"):
                    hits = memory.get_recent_memories(n=8)
                else:
                    hits = []
                for hit in hits or []:
                    knowledge.append(self._normalise_memory(hit))
            except Exception:
                pass

        if not knowledge and request.payload:
            knowledge.append({"source": "payload", "content": request.payload})

        simulator = self.simulator
        if simulator is not None and hasattr(simulator, "introspect"):
            try:
                insight = simulator.introspect(query)
                if insight:
                    knowledge.append({"source": "simulator", "content": insight})
            except Exception:
                pass

        return knowledge

    def _run_trials(
        self, request: SkillRequest, knowledge: List[Dict[str, Any]]
    ) -> Tuple[List[SkillTrial], int]:
        trials: List[SkillTrial] = []
        successes = 0
        requirements = request.requirements or _unique_keywords(request.description)

        for index in range(self.min_trials):
            coverage, coverage_evidence = self._coverage(knowledge, requirements)
            practice = self._simulate_practice(
                request,
                knowledge,
                requirements,
                index=index,
                coverage=coverage,
            )
            evidence = list(coverage_evidence)
            evidence.extend(practice.get("evidence", []))
            trial_success = bool(practice.get("success"))
            trials.append(
                SkillTrial(
                    index=index,
                    coverage=coverage,
                    success=trial_success,
                    evidence=evidence[:10],
                    mode=str(practice.get("mode", "coverage")),
                    summary=practice.get("summary"),
                    feedback=practice.get("feedback"),
                )
            )
            if trial_success:
                successes += 1

        return trials, successes

    def _coverage(
        self, knowledge: Iterable[Dict[str, Any]], requirements: Iterable[str]
    ) -> Tuple[float, List[str]]:
        req = [r.lower() for r in requirements if r]
        if not req:
            return 1.0, []
        if not knowledge:
            return 0.0, []

        hits = 0
        evidence: List[str] = []
        for item in knowledge:
            text = _normalise_text(item)
            low = text.lower()
            matched = [r for r in req if r in low]
            if matched:
                evidence.extend(matched)
                hits += len(set(matched))
        coverage = hits / max(1, len(req))
        coverage = max(0.0, min(1.0, coverage))
        return coverage, evidence[:10]

    def _simulate_practice(
        self,
        request: SkillRequest,
        knowledge: List[Dict[str, Any]],
        requirements: List[str],
        *,
        index: int,
        coverage: float,
    ) -> Dict[str, Any]:
        payload_snapshot = {
            "action_type": request.action_type,
            "description": request.description,
            "requirements": requirements,
            "knowledge": knowledge,
            "payload": request.payload,
            "attempt": index,
        }

        simulator = self.simulator
        if simulator is not None and hasattr(simulator, "run"):
            try:
                query = {"mode": "skill_practice", **json_sanitize(payload_snapshot)}
                result = simulator.run(query)
            except Exception:
                result = None
            practice = self._normalise_practice_result(result, coverage)
            if practice is not None:
                self._record_practice_attempt(request, index, practice, coverage, origin="simulator")
                return practice

        language = self.language
        practice = self._language_practice(language, request, payload_snapshot, coverage)
        if practice is not None:
            self._record_practice_attempt(request, index, practice, coverage, origin="language")
            return practice

        success = coverage >= self.success_threshold or coverage >= 0.6
        fallback = {
            "success": success,
            "mode": "coverage",
            "summary": "Validation basée sur la couverture des connaissances.",
            "feedback": None,
            "evidence": [],
        }
        self._record_practice_attempt(request, index, fallback, coverage, origin="coverage")
        return fallback

    def _normalise_practice_result(self, result: Any, coverage: float) -> Optional[Dict[str, Any]]:
        if result is None:
            return None

        success_value: Optional[bool] = None
        for key in ("success", "ok", "supported", "passed"):
            value = getattr(result, key, None)
            if isinstance(result, dict):
                value = result.get(key, value)
            coerced = _coerce_bool(value)
            if coerced is not None:
                success_value = coerced
                break

        if success_value is None:
            score = getattr(result, "score", None)
            if isinstance(result, dict):
                score = result.get("score", score)
            if isinstance(score, (int, float)):
                success_value = score >= self.success_threshold

        evidence: List[str] = []
        raw_evidence = getattr(result, "evidence", None)
        if isinstance(result, dict):
            raw_evidence = result.get("evidence", raw_evidence)
        if raw_evidence:
            if isinstance(raw_evidence, (list, tuple, set)):
                evidence = [_normalise_text(item) for item in raw_evidence][:10]
            else:
                evidence = [_normalise_text(raw_evidence)]

        summary = _first_text(result, "summary", "description", "result", "message")
        feedback = _first_text(result, "feedback", "notes", "comment", "analysis")
        if summary is None:
            summary = feedback
        if summary is None:
            summary = "Simulation d'entraînement réalisée."

        if success_value is None:
            success_value = coverage >= self.success_threshold or coverage >= 0.6

        return {
            "success": success_value,
            "mode": "simulator",
            "summary": summary,
            "feedback": feedback,
            "evidence": evidence,
        }

    def _language_practice(
        self,
        language: Any,
        request: SkillRequest,
        payload_snapshot: Dict[str, Any],
        coverage: float,
    ) -> Optional[Dict[str, Any]]:
        if language is None:
            return None

        attempt_summary = json_sanitize(payload_snapshot)
        response: Optional[str] = None

        try:
            if hasattr(language, "evaluate_skill_attempt"):
                response = language.evaluate_skill_attempt(attempt_summary)
            elif hasattr(language, "practice"):
                response = language.practice(attempt_summary)
            elif hasattr(language, "generate_reflective_reply"):
                arch = self.arch_ref() if self.arch_ref else None
                response = language.generate_reflective_reply(
                    arch,
                    "Évalue objectivement si la compétence peut être exercée avec succès : "
                    + json.dumps(attempt_summary, ensure_ascii=False),
                )
            elif hasattr(language, "reply"):
                response = language.reply(
                    intent="assess_skill_candidate",
                    data={
                        "skill": request.action_type,
                        "description": request.description,
                        "requirements": payload_snapshot.get("requirements", []),
                        "knowledge": payload_snapshot.get("knowledge", []),
                        "attempt": payload_snapshot.get("attempt"),
                    },
                    pragmatic={
                        "speech_act": "assessment",
                        "context": {"channel": "skill_sandbox"},
                    },
                )
        except Exception:
            response = None

        if not response:
            return None

        success = self._interpret_text_success(response, coverage)
        evidence = _unique_keywords(response)[:10]
        return {
            "success": success,
            "mode": "language",
            "summary": response,
            "feedback": response,
            "evidence": evidence,
        }

    def _record_practice_attempt(
        self,
        request: SkillRequest,
        index: int,
        result: Dict[str, Any],
        coverage: float,
        *,
        origin: str,
    ) -> None:
        metadata = {
            "action_type": request.action_type,
            "attempt_index": index,
            "mode": result.get("mode", origin),
            "origin": origin,
            "success": bool(result.get("success")),
            "coverage": float(coverage),
            "summary": (result.get("summary") or "")[:240],
        }
        self._memorise_event("skill_practice_attempt", metadata)

    def _interpret_text_success(self, text: str, coverage: float) -> bool:
        normalized = text.lower()
        for marker in self._negative_markers:
            if marker in normalized:
                return False
        for marker in self._positive_markers:
            if marker in normalized:
                return True
        return coverage >= self.success_threshold or coverage >= 0.6

    def _extract_requirements(self, request: SkillRequest) -> List[str]:
        payload = request.payload or {}
        req: List[str] = []
        raw = payload.get("requirements") or payload.get("knowledge")
        if isinstance(raw, str):
            req.extend(_unique_keywords(raw))
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                req.extend(_unique_keywords(_normalise_text(item)))
        if not req:
            req.extend(_unique_keywords(request.description))
        return req[:20]

    def _build_live_handler(
        self, action_type: str, *, interface: Optional[Any] = None
    ) -> Optional[Any]:
        manager_ref = weakref.ref(self)

        def _handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            manager = manager_ref()
            if manager is None:
                return {"ok": False, "reason": "skill_manager_unavailable"}
            return manager.execute(action_type, payload)

        return _handler

    def _render_execution_summary(
        self,
        request: SkillRequest,
        payload: Dict[str, Any],
        knowledge: List[Dict[str, Any]],
    ) -> str:
        language = self.language
        description = request.description or request.action_type.replace("_", " ")
        hints = [k.get("content") for k in knowledge if isinstance(k, dict) and "content" in k]
        hints_text = ", ".join(str(h)[:120] for h in hints[:4])
        message = f"Execution synthèse pour {description}."  # fallback

        if language is not None:
            prompt = {
                "topic": description,
                "summary": hints_text or "Synthèse des connaissances intégrées.",
                "payload": payload,
            }
            try:
                if hasattr(language, "reply"):
                    message = language.reply(
                        intent="inform",
                        data={"topic": description, "summary": hints_text, "payload": payload},
                        pragmatic={"speech_act": "statement", "context": {"tone": "confident"}},
                    )
                elif hasattr(language, "generate_reflective_reply"):
                    arch = self.arch_ref() if self.arch_ref else None
                    message = language.generate_reflective_reply(
                        arch,
                        f"Synthétise l'action {description} avec ces éléments: {json.dumps(json_sanitize(prompt))}",
                    )
            except Exception:
                pass

        return message

    def _memorise_event(self, kind: str, metadata: Dict[str, Any]) -> None:
        memory = self.memory
        if memory is None or not hasattr(memory, "add_memory"):
            return
        try:
            memory.add_memory({"kind": kind, "content": metadata.get("action_type", kind), "metadata": metadata})
        except Exception:
            pass

    def _notify_ready(self, request: SkillRequest) -> None:
        if request.status == "awaiting_approval":
            self._memorise_event(
                "skill_ready_for_review",
                {
                    "action_type": request.action_type,
                    "requirements": request.requirements,
                    "attempts": request.attempts,
                    "success_rate": self._success_rate(request),
                },
            )
        elif request.status == "active":
            self._memorise_event(
                "skill_auto_activated",
                {
                    "action_type": request.action_type,
                    "success_rate": self._success_rate(request),
                },
            )

    def _normalise_memory(self, hit: Any) -> Dict[str, Any]:
        if isinstance(hit, dict):
            return hit
        try:
            if hasattr(hit, "to_dict"):
                return hit.to_dict()
        except Exception:
            pass
        try:
            return json_sanitize(hit)  # type: ignore[arg-type]
        except Exception:
            return {"content": _normalise_text(hit)}

    def _success_rate(self, request: SkillRequest) -> float:
        total = len(request.trials)
        if total <= 0:
            return 0.0
        wins = sum(1 for trial in request.trials if trial.success)
        return wins / total

    # ------------------------------------------------------------------
    # Persistence
    def _save_state_locked(self) -> None:
        data = {
            action_type: {
                "identifier": req.identifier,
                "action_type": req.action_type,
                "description": req.description,
                "payload": req.payload,
                "created_at": req.created_at,
                "status": req.status,
                "attempts": req.attempts,
                "successes": req.successes,
                "knowledge": req.knowledge,
                "requirements": req.requirements,
                "approval_required": req.approval_required,
                "approved_by": req.approved_by,
                "approval_notes": req.approval_notes,
                "last_error": req.last_error,
                "trials": [
                    {
                        "index": t.index,
                        "coverage": t.coverage,
                        "success": t.success,
                        "evidence": list(t.evidence),
                        "mode": t.mode,
                        "summary": t.summary,
                        "feedback": t.feedback,
                    }
                    for t in req.trials
                ],
            }
            for action_type, req in self._requests.items()
        }
        try:
            with open(self.index_path, "w", encoding="utf-8") as handle:
                json.dump(json_sanitize(data), handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for action_type, data in raw.items():
            try:
                req = SkillRequest(
                    identifier=str(data.get("identifier", uuid.uuid4())),
                    action_type=action_type,
                    description=str(data.get("description", action_type)),
                    payload=dict(data.get("payload", {})),
                    created_at=float(data.get("created_at", _now())),
                    status=str(data.get("status", "pending")),
                    attempts=int(data.get("attempts", 0)),
                    successes=int(data.get("successes", 0)),
                    approval_required=bool(data.get("approval_required", self.approval_required_default)),
                )
                req.knowledge = list(data.get("knowledge", []))
                req.requirements = list(data.get("requirements", []))
                req.approved_by = data.get("approved_by")
                req.approval_notes = data.get("approval_notes")
                req.last_error = data.get("last_error")
                req.trials = [
                    SkillTrial(
                        index=int(t.get("index", 0)),
                        coverage=float(t.get("coverage", 0.0)),
                        success=bool(t.get("success", False)),
                        evidence=list(t.get("evidence", [])),
                        mode=str(t.get("mode", "coverage")),
                        summary=t.get("summary"),
                        feedback=t.get("feedback"),
                    )
                    for t in data.get("trials", [])
                ]
                if not req.requirements:
                    req.requirements = self._extract_requirements(req)
                self._requests[action_type] = req
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Utility
    def __len__(self) -> int:
        with self._lock:
            return len(self._requests)

