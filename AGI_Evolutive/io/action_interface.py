"""Action interface bridging goal planning and concrete execution."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from AGI_Evolutive.beliefs.graph import Evidence
from AGI_Evolutive.utils.jsonsafe import json_sanitize


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
                "update_belief": lambda act: self._h_update_belief(act.payload, act.context),
                "assert_fact": lambda act: self._h_assert_fact(act.payload, act.context),
                "link_entity": lambda act: self._h_link_entity(act.payload, act.context),
                "abduce": lambda act: self._h_abduce(act.payload, act.context),
                "set_user_pref": lambda act: self._h_set_user_pref(act.payload, act.context),
                "self_improve": lambda act: self._h_self_improve(act.payload, act.context),
                "promote": lambda act: self._h_promote(act.payload, act.context),
                "rollback": lambda act: self._h_rollback(act.payload, act.context),
                "simulate": self._h_simulate,
                "plan": self._h_plan,
                "simulate_dialogue": self._h_simulate_dialogue,
                "search_counterexample": self._h_search_counterexample,
                "ask_clarifying": lambda act: self._h_ask(act.payload, act.context),
                "code_evolve": lambda act: self._h_code_evolve(act.payload, act.context),
                "promote_code": lambda act: self._h_promote_code(act.payload, act.context),
                "rollback_code": lambda act: self._h_rollback_code(act.payload, act.context),
                "rotate_curriculum": lambda act: self._h_rotate_curriculum(act.payload, act.context),
            }
            handler = handlers.get(act.type, self._h_simulate)

            act.status = "running"
            try:
                if act.type == "learn_concept":
                    result = self._h_learn_concept(act.payload, act.context)
                else:
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
            payload = {"kind": kind, "content": content, "metadata": meta}
            mem_id = memory.add_memory(payload)
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

    def _h_learn_concept(self, payload: Dict[str, Any], context: Dict[str, Any]):
        concept = (payload or {}).get("concept")
        why = (payload or {}).get("why") or (payload or {}).get("reason") or "learning_goal"
        if not concept:
            return {"ok": False, "error": "no concept"}

        memory = self.bound.get("memory") if hasattr(self, "bound") else None
        # trace d'intention
        try:
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    {
                        "kind": "learning_intent",
                        "content": f"Apprendre le concept : {concept}",
                        "metadata": {"reason": why, "source": "action_interface"},
                    }
                )
        except Exception:
            pass

        # auto-évaluation
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        learn = getattr(arch, "learning", None) if arch else None
        try:
            result = learn.self_assess_concept(concept) if learn and hasattr(learn, "self_assess_concept") else {"confidence": 0.0}
        except Exception:
            result = {"confidence": 0.0}
        conf = float(result.get("confidence", 0.0))

        # intégration auto si confiance élevée
        if conf >= 0.90 and arch and hasattr(arch, "_record_skill"):
            try:
                arch._record_skill(concept)
                if memory and hasattr(memory, "add_memory"):
                    memory.add_memory(
                        {
                            "kind": "learning_validated",
                            "content": concept,
                            "metadata": {"source": "self_assess", "confidence": conf},
                        }
                    )
                return {"ok": True, "concept": concept, "integrated": True, "confidence": conf}
            except Exception:
                pass

        # sinon, demande de validation à l'utilisateur
        try:
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    {
                        "kind": "validation_request",
                        "content": f"Valider mon apprentissage du concept: {concept}",
                        "metadata": {
                            "concept": concept,
                            "need": "confirm_understanding",
                            "confidence": conf,
                        },
                    }
                )
        except Exception:
            pass
        return {"ok": True, "concept": concept, "integrated": False, "confidence": conf}

    def _h_update_belief(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        if not arch or not hasattr(arch, "beliefs"):
            return {"ok": False, "error": "beliefs not available"}
        payload = payload or {}
        subject = payload.get("subject")
        relation = payload.get("relation")
        value = payload.get("value")
        conf = float(payload.get("confidence", 0.5))
        pol = int(payload.get("polarity", +1))
        ev = Evidence.new(
            kind=payload.get("kind", "reasoning"),
            source=payload.get("source", "self"),
            snippet=payload.get("snippet", ""),
            weight=float(payload.get("weight", 0.5)),
        )
        b = arch.beliefs.upsert(
            subject,
            relation,
            value,
            confidence=conf,
            polarity=pol,
            evidence=ev,
            created_by="action_interface",
        )
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="belief_update",
                    content=f"{subject} {relation} {value}",
                    metadata={"conf": b.confidence, "pol": b.polarity},
                )
        except Exception:
            pass
        return {"ok": True, "belief": {"id": b.id, "conf": b.confidence}}

    def _h_assert_fact(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        if not arch or not hasattr(arch, "beliefs"):
            return {"ok": False, "error": "beliefs_unavailable"}
        payload = payload or {}
        subject = payload.get("subject")
        relation = payload.get("relation")
        value = payload.get("value")
        if not all([subject, relation, value]):
            return {"ok": False, "error": "missing_fact_fields"}
        confidence = float(payload.get("confidence", 0.6))
        polarity = int(payload.get("polarity", 1))
        evidence_text = payload.get("evidence") or (context or {}).get("evidence") or f"{subject} {relation} {value}"
        ev = Evidence.new("action", "assert_fact", evidence_text, weight=min(1.0, max(0.0, confidence)))
        belief = arch.beliefs.upsert(
            subject,
            relation,
            value,
            confidence=confidence,
            polarity=polarity,
            evidence=ev,
            created_by="action_interface",
        )
        try:
            scm = getattr(arch, "scm", None)
            if scm and hasattr(scm, "refresh_from_belief"):
                scm.refresh_from_belief(belief)
        except Exception:
            pass
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="belief_update",
                    content=f"{belief.subject} {belief.relation} {belief.value}",
                    metadata={"conf": belief.confidence, "pol": belief.polarity},
                )
        except Exception:
            pass
        contradictions: List[Dict[str, Any]] = []
        try:
            for positive, negative in arch.beliefs.find_contradictions(min_conf=0.6):
                if (
                    positive.subject == belief.subject
                    and positive.relation == belief.relation
                    and positive.value == belief.value
                ):
                    contradictions.append({"positive": positive.id, "negative": negative.id})
                    memory = self.bound.get("memory") if hasattr(self, "bound") else None
                    if memory and hasattr(memory, "add_memory"):
                        memory.add_memory(
                            kind="contradiction_detected",
                            content=f"{positive.subject} {positive.relation}",
                            metadata={"positive": positive.id, "negative": negative.id},
                        )
        except Exception:
            pass
        return {"ok": True, "belief_id": belief.id, "contradictions": contradictions}

    def _h_link_entity(self, payload: Dict[str, Any], context: Dict[str, Any]):
        payload = payload or {}
        text = payload.get("text")
        if not text:
            return {"ok": False, "error": "missing_text"}
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        linker = getattr(arch, "entity_linker", None) if arch else None
        if not linker:
            return {"ok": False, "error": "linker_unavailable"}
        result = linker.link(text, hint_type=payload.get("type"))
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="entity_resolved",
                    content=text,
                    metadata=result,
                )
        except Exception:
            pass
        return {"ok": True, "entity": result}

    def _h_abduce(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        if not arch or not hasattr(arch, "abduction"):
            return {"ok": False, "error": "abduction not available"}
        payload = payload or {}
        context = context or {}
        obs = payload.get("observation") or context.get("observation") or ""
        hyps = arch.abduction.generate(obs)
        if not hyps:
            return {"ok": True, "hypotheses": []}
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="hypothesis",
                    content=hyps[0].label,
                    metadata={"score": hyps[0].score},
                )
        except Exception:
            pass
        return {"ok": True, "hypotheses": [h.__dict__ for h in hyps]}

    def _h_set_user_pref(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch") if hasattr(self, "bound") else None
        if not arch or not hasattr(arch, "user_model"):
            return {"ok": False, "error": "user_model not available"}
        payload = payload or {}
        label = payload.get("label")
        liked = bool(payload.get("liked", True))
        arch.user_model.observe_preference(str(label), liked)
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="user_pref",
                    content=("like:" if liked else "dislike:") + str(label),
                    metadata={"source": "action_interface"},
                )
        except Exception:
            pass
        return {"ok": True}

    def _h_self_improve(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        if not arch or not hasattr(arch, "self_improver"):
            return {"ok": False, "error": "self_improver not available"}
        params = payload or {}
        n = int(params.get("n", 4))
        cid = arch.self_improver.run_cycle(n_candidates=n)
        return {"ok": True, "candidate_id": cid}

    def _h_promote(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        if not arch or not hasattr(arch, "self_improver"):
            return {"ok": False, "error": "self_improver not available"}
        cid = (payload or {}).get("cid")
        if not cid:
            return {"ok": False, "error": "missing cid"}
        arch.self_improver.promote(cid)
        return {"ok": True}

    def _h_rollback(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        if not arch or not hasattr(arch, "self_improver"):
            return {"ok": False, "error": "self_improver not available"}
        params = payload or {}
        steps = int(params.get("steps", 1))
        arch.self_improver.rollback(steps=steps)
        return {"ok": True}

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
        arch = self.bound.get("arch")
        if not arch or not hasattr(arch, "simulator"):
            return {"ok": False, "error": "simulator_unavailable"}
        query = dict(act.payload or {})
        try:
            report = arch.simulator.run(query)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="counterfactual_result",
                    content=str(query)[:160],
                    metadata={"supported": report.supported},
                )
        except Exception:
            pass
        return {
            "ok": True,
            "supported": report.supported,
            "evidence": report.evidence,
            "intervention": report.intervention,
            "simulations": report.simulations,
        }

    def _h_simulate_dialogue(self, act: Action) -> Dict[str, Any]:
        payload = dict(act.payload or {})
        context = dict(act.context or {})
        rule_id = payload.get("rule_id") or context.get("rule_id")
        memory = self.bound.get("memory") if hasattr(self, "bound") else None
        try:
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory({"kind": "sim_result", "rule_id": rule_id, "ok": True})
        except Exception:
            pass
        return {"ok": True, "rule_id": rule_id, "simulated": True}

    def _h_search_counterexample(self, act: Action) -> Dict[str, Any]:
        payload = dict(act.payload or {})
        context = dict(act.context or {})
        rule_id = payload.get("rule_id") or context.get("rule_id")
        memory = self.bound.get("memory") if hasattr(self, "bound") else None
        try:
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory({"kind": "counterexample_scan", "rule_id": rule_id, "found": False})
        except Exception:
            pass
        return {"ok": True, "rule_id": rule_id, "found": False}

    def _h_plan(self, act: Action) -> Dict[str, Any]:
        arch = self.bound.get("arch")
        if not arch or not hasattr(arch, "planner"):
            return {"ok": False, "error": "planner_unavailable"}
        goal = act.payload.get("goal") or act.payload.get("text") or "objectif"
        steps = arch.planner.plan("diagnostic_general", context={"goal": goal})
        if not steps:
            steps = [
                f"Clarifier le résultat pour « {goal} ».",
                "Lister les ressources nécessaires.",
                "Programmer une première action concrète.",
            ]
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="plan_created",
                    content=goal[:160],
                    metadata={"steps": steps},
                )
        except Exception:
            pass
        return {"ok": True, "goal": goal, "steps": steps}

    def _h_ask(self, payload: Dict[str, Any], context: Dict[str, Any]):
        payload = payload or {}
        question = payload.get("question") or payload.get("text")
        if not question:
            return {"ok": False, "error": "missing_question"}
        arch = self.bound.get("arch")
        qm = getattr(arch, "question_manager", None) if arch else None
        if qm and hasattr(qm, "add_question"):
            qm.add_question(question, qtype=payload.get("type", "clarifying"))
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="question_active",
                    content=question,
                    metadata={"source": payload.get("source", "action")},
                )
        except Exception:
            pass
        return {"ok": True, "question": question}

    def _h_code_evolve(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        improver = getattr(arch, "self_improver", None)
        if not improver or not hasattr(improver, "run_code_cycle"):
            return {"ok": False, "error": "code_evolver_unavailable"}
        payload = payload or {}
        cid = improver.run_code_cycle(n_candidates=int(payload.get("n", 2)))
        return {"ok": True, "candidate_id": cid}

    def _h_promote_code(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        prom = getattr(arch, "promotions", None)
        if not prom:
            return {"ok": False, "error": "promotions_unavailable"}
        cid = (payload or {}).get("cid")
        if not cid:
            return {"ok": False, "error": "missing cid"}
        metadata: Dict[str, Any] = {}
        try:
            candidate = prom.read_candidate(cid)
            metadata = candidate.get("metadata", {}) or {}
            patch_payload = metadata.get("patch")
            code_evolver = getattr(arch, "code_evolver", None)
            if patch_payload and code_evolver:
                code_evolver.promote_patch(patch_payload)
        except Exception:
            metadata = metadata or {}
        prom.promote(cid)
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="promotion_code",
                    content=str(cid),
                    metadata=metadata,
                )
        except Exception:
            pass
        return {"ok": True}

    def _h_rollback_code(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        prom = getattr(arch, "promotions", None)
        if not prom:
            return {"ok": False, "error": "promotions_unavailable"}
        steps = int((payload or {}).get("steps", 1))
        prom.rollback(steps=steps)
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    kind="rollback_code",
                    content=f"rollback {steps}",
                    metadata={},
                )
        except Exception:
            pass
        return {"ok": True}

    def _h_rotate_curriculum(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        improver = getattr(arch, "self_improver", None)
        if not improver or not hasattr(improver, "rotate_curriculum"):
            return {"ok": False, "error": "self_improver_unavailable"}
        level = str((payload or {}).get("level", "base"))
        cid = improver.rotate_curriculum(level)
        return {"ok": True, "candidate_id": cid, "level": level}

    # ------------------------------------------------------------------
    # Logging & memory
    def _log(self, act: Action, reward: float = 0.0) -> None:
        rec = asdict(act)
        rec["reward"] = reward
        rec["logged_at"] = _now()
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(rec), ensure_ascii=False) + "\n")

    def _memorize_action(self, act: Action, reward: float = 0.0) -> None:
        memory = self.bound.get("memory")
        if not (memory and hasattr(memory, "add_memory")):
            return
        try:
            memory.add_memory(
                {
                    "kind": "action_experience",
                    "content": f"[{act.type}] -> {act.status}",
                    "metadata": {
                        "action_id": act.id,
                        "payload": act.payload,
                        "result": act.result,
                        "reward": reward,
                        "priority": act.priority,
                        "created_at": act.created_at,
                    },
                }
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
