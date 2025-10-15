"""Action interface bridging goal planning and concrete execution."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


def _now() -> float:
    return time.time()


@dataclass
class Action:
    id: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    created_at: float = field(default_factory=_now)
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ActionInterface:
    """Unified action execution layer with backward compatibility helpers."""

    def __init__(
        self,
        memory_store: Optional[Any] = None,
        path_log: str = "data/actions_log.jsonl",
        output_dir: str = "data/output",
    ) -> None:
        os.makedirs(os.path.dirname(path_log), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.path_log = path_log
        self.output_dir = output_dir

        self.bound: Dict[str, Any] = {
            "arch": None,
            "goals": None,
            "policy": None,
            "memory": memory_store,
            "metacog": None,
            "emotions": None,
            "language": None,
        }

        self.queue: List[Action] = []
        self.cooldown_s = 0.0
        self._legacy_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Binding helpers
    def bind(
        self,
        arch: Any = None,
        goals: Any = None,
        policy: Any = None,
        memory: Any = None,
        metacog: Any = None,
        emotions: Any = None,
        language: Any = None,
    ) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        self.bound.update(
            {
                "arch": arch,
                "goals": goals,
                "policy": policy,
                "metacog": metacog,
                "emotions": emotions,
                "language": language,
            }
        )

    # ------------------------------------------------------------------
    # Registration helpers
    def register_handler(self, action_type: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Compatibility hook for legacy direct handlers."""

        self._legacy_handlers[action_type] = fn

    # ------------------------------------------------------------------
    # Queue management
    def enqueue(
        self,
        type_: str,
        payload: Dict[str, Any],
        priority: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        act = Action(
            id=str(uuid.uuid4()),
            type=type_,
            payload=payload or {},
            priority=float(priority),
            context=context or {},
        )
        self.queue.append(act)
        self.queue.sort(key=lambda a: a.priority, reverse=True)
        return act.id

    def _pull_from_goals(self) -> Optional[Action]:
        goals = self.bound.get("goals")
        try:
            if goals and hasattr(goals, "pop_next_action"):
                nxt = goals.pop_next_action()
                if nxt:
                    return Action(
                        id=str(uuid.uuid4()),
                        type=nxt.get("type", "reflect"),
                        payload=nxt.get("payload", {}),
                        priority=float(nxt.get("priority", 0.5)),
                        context={"source": "goals"},
                    )
            elif goals and hasattr(goals, "get_next_action"):
                nxt = goals.get_next_action()
                if nxt:
                    return Action(
                        id=str(uuid.uuid4()),
                        type=nxt.get("type", "reflect"),
                        payload=nxt.get("payload", {}),
                        priority=float(nxt.get("priority", 0.5)),
                        context={"source": "goals"},
                    )
        except Exception:
            pass
        return None

    def step(self) -> None:
        emo = self.bound.get("emotions")
        if emo and hasattr(emo, "get_emotional_modulators"):
            mods = emo.get_emotional_modulators() or {}
            exploration = float(mods.get("exploration_rate", 0.15))
            self.cooldown_s = max(0.0, 1.0 - 0.8 * exploration)
        else:
            self.cooldown_s = 0.5

        if not self.queue:
            pulled = self._pull_from_goals()
            if pulled:
                self.queue.append(pulled)

        if not self.queue:
            self._maybe_autonomous_microaction()

        if not self.queue:
            return

        act = self.queue.pop(0)
        self._execute(act)
        time.sleep(self.cooldown_s)

    # ------------------------------------------------------------------
    # Direct execution compatibility
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = action.get("payload")
        payload = dict(payload) if isinstance(payload, dict) else {}
        for key, value in action.items():
            if key not in {"type", "payload", "priority", "context", "id"}:
                payload.setdefault(key, value)

        act = Action(
            id=str(action.get("id", str(uuid.uuid4()))),
            type=(action.get("type") or "simulate"),
            payload=payload,
            priority=float(action.get("priority", 0.5)),
            context=dict(action.get("context", {})),
        )
        self._execute(act)
        return act.result or {"ok": False, "reason": "no_result"}

    # ------------------------------------------------------------------
    # Execution core
    def _execute(self, act: Action) -> None:
        policy = self.bound.get("policy")
        arch = self.bound.get("arch")
        memory = self.bound.get("memory")
        metacog = self.bound.get("metacog")
        emotions = self.bound.get("emotions")

        if policy and hasattr(policy, "validate_action"):
            try:
                ok, reason = policy.validate_action(asdict(act))
                if not ok:
                    act.status = "skipped"
                    act.result = {"ok": False, "reason": reason or "policy_rejected"}
                    self._log(act)
                    self._memorize_action(act)
                    return
            except Exception:
                pass

        if act.type in self._legacy_handlers:
            try:
                payload = {"type": act.type, **act.payload}
                result = self._legacy_handlers[act.type](payload)
                act.status = "done" if result.get("ok", True) else "failed"
                act.result = result
            except Exception as e:
                act.status = "failed"
                act.result = {"ok": False, "error": str(e)}
        else:
            handlers = {
                "message_user": self._h_message_user,
                "write_memory": self._h_write_memory,
                "save_file": self._h_save_file,
                "reflect": self._h_reflect,
                "learn_concept": self._h_learn_concept,
                "search_memory": self._h_search_memory,
            }
            handler = handlers.get(act.type, self._h_simulate)

            act.status = "running"
            try:
                result = handler(act)
                success = not isinstance(result, dict) or result.get("ok", True)
                act.status = "done" if success else "failed"
                act.result = result if isinstance(result, dict) else {"ok": True, "data": result}
            except Exception as e:
                act.status = "failed"
                act.result = {"ok": False, "error": str(e)}

        reward = self._shape_reward(act)
        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(
                    kind="action_success" if act.status == "done" else "action_failure",
                    intensity=0.4 if act.status == "done" else 0.6,
                    valence_hint=+0.4 if act.status == "done" else -0.5,
                    arousal_hint=0.2,
                    meta={"action_type": act.type},
                )
            except Exception:
                pass

        if metacog and hasattr(metacog, "_record_metacognitive_event"):
            try:
                metacog._record_metacognitive_event(
                    event_type="action_executed",
                    domain=getattr(metacog, "CognitiveDomain", None),
                    description=f"Action {act.type} -> {act.status}",
                    significance=0.3 if act.status == "done" else 0.5,
                    confidence=0.7,
                    emotional_valence=+0.3 if act.status == "done" else -0.2,
                    cognitive_load=0.2,
                )
            except Exception:
                pass

        self._log(act, reward=reward)
        self._memorize_action(act, reward=reward)

    # ------------------------------------------------------------------
    # Handlers
    def _h_message_user(self, act: Action) -> Dict[str, Any]:
        lang = self.bound.get("language")
        text = act.payload.get("text")
        if not text and lang and hasattr(lang, "generate"):
            intent = act.payload.get("intent", "inform")
            text = lang.generate({"intent": intent, "hints": getattr(lang, "style_hints", {})})
        text = text or "(message vide)"

        path = os.path.join(self.output_dir, "last_message.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": path, "text": text}

    def _h_write_memory(self, act: Action) -> Dict[str, Any]:
        memory = self.bound.get("memory")
        if not (memory and hasattr(memory, "add_memory")):
            return {"ok": False, "reason": "memory_unavailable"}
        kind = act.payload.get("kind", "note")
        content = act.payload.get("content", "")
        meta = act.payload.get("meta", {})
        try:
            mem_id = memory.add_memory(kind=kind, content=content, metadata=meta)
            return {"ok": True, "memory_id": mem_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_save_file(self, act: Action) -> Dict[str, Any]:
        name = act.payload.get("name", f"artifact_{int(time.time())}.txt")
        content = act.payload.get("content", "")
        path = os.path.join(self.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": path}

    def _h_reflect(self, act: Action) -> Dict[str, Any]:
        metacog = self.bound.get("metacog")
        if not (metacog and hasattr(metacog, "trigger_reflection")):
            return {"ok": False, "reason": "metacog_unavailable"}
        trigger = act.payload.get("trigger", "action_reflect")
        domain = getattr(getattr(metacog, "CognitiveDomain", None), "REASONING", None)
        try:
            ref = metacog.trigger_reflection(trigger=trigger, domain=domain, urgency=0.4, depth=2)
            return {"ok": True, "reflection": {"duration": getattr(ref, "duration", None), "quality": getattr(ref, "quality_score", None)}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_learn_concept(self, act: Action) -> Dict[str, Any]:
        goals = self.bound.get("goals")
        memory = self.bound.get("memory")
        concept = act.payload.get("concept", "concept_inconnu")
        why = act.payload.get("why", "lacune détectée")
        try:
            if goals and hasattr(goals, "add_learning_goal"):
                goals.add_learning_goal(concept, motive=why, expected_gain=0.4)
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="learning_intent",
                    content=f"Apprendre le concept: {concept}",
                    metadata={"reason": why, "source": "action_interface"},
                )
            return {"ok": True, "concept": concept}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_search_memory(self, act: Action) -> Dict[str, Any]:
        memory = self.bound.get("memory")
        query = act.payload.get("query", "")
        if not query:
            return {"ok": False, "reason": "empty_query"}
        try:
            if hasattr(memory, "search"):
                hits = memory.search(query, top_k=10)
            else:
                hits = memory.get_recent_memories(n=50) if memory else []
                hits = [m for m in hits if query.lower() in str(m.get("content", "")).lower()]
            return {"ok": True, "hits": hits[:10]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _h_simulate(self, act: Action) -> Dict[str, Any]:
        ok = (time.time() * 1000) % 10 != 0
        return {"ok": bool(ok), "simulated": True, "type": act.type}

    # ------------------------------------------------------------------
    # Logging & memory
    def _log(self, act: Action, reward: float = 0.0) -> None:
        rec = asdict(act)
        rec["reward"] = reward
        rec["logged_at"] = _now()
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _memorize_action(self, act: Action, reward: float = 0.0) -> None:
        memory = self.bound.get("memory")
        if not (memory and hasattr(memory, "add_memory")):
            return
        try:
            memory.add_memory(
                kind="action_experience",
                content=f"[{act.type}] -> {act.status}",
                metadata={
                    "action_id": act.id,
                    "payload": act.payload,
                    "result": act.result,
                    "reward": reward,
                    "priority": act.priority,
                    "created_at": act.created_at,
                },
            )
        except Exception:
            pass

    def _shape_reward(self, act: Action) -> float:
        base = 0.6 if act.status == "done" else -0.5
        util = 0.2 if act.type in ("learn_concept", "reflect", "write_memory") else 0.0
        return float(max(-1.0, min(1.0, base + util)))

    def _maybe_autonomous_microaction(self) -> None:
        emo = self.bound.get("emotions")
        curiosity = 0.2
        if emo and hasattr(emo, "get_emotional_modulators"):
            mods = emo.get_emotional_modulators() or {}
            curiosity = float(mods.get("curiosity_gain", 0.2)) + float(mods.get("exploration_rate", 0.15))
        if curiosity > 0.25:
            if int(time.time()) % 2 == 0:
                self.enqueue("reflect", {"trigger": "idle_reflection"}, priority=0.55, context={"auto": True})
            else:
                self.enqueue("search_memory", {"query": "lacune|erreur|incompréhension"}, priority=0.52, context={"auto": True})
