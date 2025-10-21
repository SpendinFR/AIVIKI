"""Action interface bridging goal planning and concrete execution."""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from AGI_Evolutive.beliefs.graph import Evidence
from AGI_Evolutive.utils.jsonsafe import json_sanitize

try:  # pragma: no cover - fallback exercised in tests when metacognition deps missing
    from AGI_Evolutive.metacognition import CognitiveDomain as _CognitiveDomain
except Exception:  # noqa: BLE001 - best-effort import with graceful degradation
    class _FallbackCognitiveDomain(Enum):
        REASONING = "reasoning"

    CognitiveDomain = _FallbackCognitiveDomain
else:
    CognitiveDomain = _CognitiveDomain


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
    meta: Dict[str, Any] = field(default_factory=dict)


class DiscreteThompsonSampling:
    """Simple Thompson Sampling helper on a discrete candidate set."""

    def __init__(self, options: Sequence[Any]) -> None:
        if not options:
            raise ValueError("options must not be empty")
        self._options: Tuple[Any, ...] = tuple(options)
        self._success: Dict[Any, float] = {opt: 1.0 for opt in self._options}
        self._failure: Dict[Any, float] = {opt: 1.0 for opt in self._options}

    def select(self) -> Any:
        best_opt = self._options[0]
        best_score = -1.0
        for opt in self._options:
            score = random.betavariate(self._success[opt], self._failure[opt])
            if score > best_score:
                best_opt = opt
                best_score = score
        return best_opt

    def update(self, option: Any, reward: float) -> None:
        if option not in self._success:
            return
        reward = max(-1.0, min(1.0, float(reward)))
        prob = 0.5 + 0.5 * reward
        self._success[option] += prob
        self._failure[option] += 1.0 - prob


class AdaptiveEMA:
    """Adaptive exponential moving average controlled by Thompson Sampling."""

    def __init__(self, betas: Sequence[float]) -> None:
        candidates = [float(b) for b in betas if 0.0 < float(b) < 1.0]
        if not candidates:
            raise ValueError("AdaptiveEMA requires betas in (0,1)")
        self.selector = DiscreteThompsonSampling(tuple(candidates))
        self.value: float = 0.5

    def step(self, observation: float, reward: float) -> Tuple[float, float, float]:
        obs = max(0.0, min(1.0, observation))
        beta = float(self.selector.select())
        updated = beta * self.value + (1.0 - beta) * obs
        drift = updated - self.value
        self.value = updated
        self.selector.update(beta, reward)
        return beta, self.value, drift


@dataclass
class ActionStats:
    ema_selector: AdaptiveEMA
    ema: float = 0.5
    weight_count: float = 0.0
    last_feedback_at: float = field(default_factory=_now)
    last_enqueue_at: float = field(default_factory=_now)
    last_reward: float = 0.0
    last_beta: float = 0.5
    drift: float = 0.0
    last_drift_log_at: float = 0.0

    def touch_enqueue(self) -> None:
        self.last_enqueue_at = _now()

    def update(self, reward: float) -> Dict[str, float]:
        scaled = max(0.0, min(1.0, 0.5 + 0.5 * reward))
        beta, ema, drift = self.ema_selector.step(scaled, reward)
        self.ema = ema
        self.weight_count = self.weight_count * 0.97 + 1.0
        now = _now()
        age = now - self.last_feedback_at
        self.last_feedback_at = now
        self.last_reward = reward
        self.last_beta = beta
        self.drift = drift
        return {"beta": beta, "drift": drift, "age": age, "ema": ema}

    def recency(self, now: float, horizon: float) -> float:
        age = max(0.0, now - self.last_feedback_at)
        if horizon <= 0:
            return 0.0
        return max(0.0, min(1.0, age / horizon))

    def drift_feature(self) -> float:
        return max(0.0, min(1.0, abs(self.drift) * 4.0))


class PriorityLearner:
    """Online GLM-style learner to adapt action priorities."""

    def __init__(self) -> None:
        self.feature_count = 5
        self.weights: List[float] = [0.0] * (self.feature_count + 1)
        self.lr = 0.08
        self.max_step = 0.05
        self.weight_clamp = 4.0
        self.recency_horizon = 120.0
        self.mix_options: Tuple[Tuple[float, float, float], ...] = (
            (0.60, 0.30, 0.10),
            (0.45, 0.45, 0.10),
            (0.35, 0.55, 0.10),
            (0.50, 0.25, 0.25),
        )
        self.mix_selector = DiscreteThompsonSampling(self.mix_options)

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def predict(self, features: Sequence[float]) -> float:
        z = self.weights[0]
        for idx, feature in enumerate(features):
            z += self.weights[idx + 1] * feature
        return self._sigmoid(z)

    def compute_priority(
        self,
        base_priority: float,
        success_score: float,
        context_signal: float,
        recency: float,
        drift: float,
    ) -> Dict[str, Any]:
        features = [
            max(0.0, min(1.0, base_priority)),
            max(0.0, min(1.0, success_score)),
            max(0.0, min(1.0, context_signal)),
            max(0.0, min(1.0, recency)),
            max(0.0, min(1.0, drift)),
        ]
        learned = self.predict(features)
        mix = self.mix_selector.select()
        base_w, learned_w, context_w = mix
        residual = max(0.0, 1.0 - (base_w + learned_w + context_w))
        priority = (
            base_w * features[0]
            + learned_w * learned
            + context_w * features[2]
            + residual * features[1]
        )
        priority = max(0.0, min(1.0, priority))
        return {
            "priority": priority,
            "features": features,
            "mix": mix,
            "learned": learned,
            "context_signal": features[2],
            "base_priority": features[0],
        }

    def update(self, info: Optional[Dict[str, Any]], reward: float) -> None:
        if not info:
            return
        features = info.get("features")
        if not features:
            return
        target = max(0.0, min(1.0, 0.5 + 0.5 * reward))
        pred = self.predict(features)
        error = pred - target
        bias_step = max(-self.max_step, min(self.max_step, self.lr * error))
        self.weights[0] -= bias_step
        self.weights[0] = max(-self.weight_clamp, min(self.weight_clamp, self.weights[0]))
        for idx, feature in enumerate(features):
            grad = self.lr * error * feature
            grad = max(-self.max_step, min(self.max_step, grad))
            self.weights[idx + 1] -= grad
            self.weights[idx + 1] = max(
                -self.weight_clamp, min(self.weight_clamp, self.weights[idx + 1])
            )
        mix = info.get("mix")
        if mix:
            self.mix_selector.update(tuple(mix), reward)
        self.lr = max(0.01, self.lr * 0.999)


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
            "simulator": None,
            "jobs": None,
        }

        self.queue: List[Action] = []
        self.cooldown_s = 0.0
        self._legacy_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.current_mode: Optional[str] = None
        self._priority_learner = PriorityLearner()
        self._action_stats: Dict[str, ActionStats] = {}
        self._last_emotion_modulators: Dict[str, float] = {}

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
        simulator: Any = None,
        jobs: Any = None,
        perception: Any = None,
    ) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        if jobs is not None:
            self.bound["jobs"] = jobs
        if perception is not None:
            self.bound["perception"] = perception
        if simulator is None and arch is not None:
            simulator = getattr(arch, "simulator", None)
        if simulator is not None:
            self.bound["simulator"] = simulator
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
        base_priority = float(priority)
        act = Action(
            id=str(uuid.uuid4()),
            type=type_,
            payload=payload or {},
            priority=base_priority,
            context=context or {},
        )
        self._prepare_priority_learning(act, base_priority, adjust_priority=True)
        self.queue.append(act)
        self.queue.sort(key=lambda a: a.priority, reverse=True)
        return act.id

    def _pull_from_goals(self) -> Optional[Action]:
        goals = self.bound.get("goals")
        try:
            if goals and hasattr(goals, "pop_next_action"):
                nxt = goals.pop_next_action()
                if nxt:
                    act = Action(
                        id=str(uuid.uuid4()),
                        type=nxt.get("type", "reflect"),
                        payload=nxt.get("payload", {}),
                        priority=float(nxt.get("priority", 0.5)),
                        context={"source": "goals"},
                    )
                    self._prepare_priority_learning(act, act.priority, adjust_priority=True)
                    return act
            elif goals and hasattr(goals, "get_next_action"):
                nxt = goals.get_next_action()
                if nxt:
                    act = Action(
                        id=str(uuid.uuid4()),
                        type=nxt.get("type", "reflect"),
                        payload=nxt.get("payload", {}),
                        priority=float(nxt.get("priority", 0.5)),
                        context={"source": "goals"},
                    )
                    self._prepare_priority_learning(act, act.priority, adjust_priority=True)
                    return act
        except Exception:
            pass
        return None

    def _get_action_stats(self, action_type: str) -> ActionStats:
        stats = self._action_stats.get(action_type)
        if stats is None:
            stats = ActionStats(AdaptiveEMA([0.2, 0.4, 0.6, 0.8]))
            self._action_stats[action_type] = stats
        return stats

    def _prepare_priority_learning(
        self, act: Action, base_priority: float, adjust_priority: bool = True
    ) -> None:
        stats = self._get_action_stats(act.type)
        now = _now()
        success_score = stats.ema_selector.value
        context_signal = self._compute_context_signal(act, stats)
        recency = stats.recency(now, self._priority_learner.recency_horizon)
        drift = stats.drift_feature()
        info = self._priority_learner.compute_priority(
            base_priority,
            success_score,
            context_signal,
            recency,
            drift,
        )
        info.update(
            {
                "action_type": act.type,
                "ema_value": stats.ema,
                "selected_beta": stats.last_beta,
                "timestamp": now,
            }
        )
        act.meta["priority_model"] = info
        if adjust_priority:
            act.priority = info["priority"]
        stats.touch_enqueue()

    def _compute_context_signal(self, act: Action, stats: ActionStats) -> float:
        urgency = float(act.context.get("urgency") or act.payload.get("urgency") or 0.0)
        urgency = max(0.0, min(1.0, urgency))
        novelty = float(act.context.get("novelty", 0.0))
        novelty = max(0.0, min(1.0, novelty))
        mods = self._last_emotion_modulators or {}
        exploration = float(mods.get("exploration_rate", 0.15))
        exploration = max(0.0, min(1.0, exploration))
        curiosity = float(mods.get("curiosity_gain", 0.15))
        curiosity = max(0.0, min(1.0, curiosity))
        load = float(mods.get("cognitive_load", 0.0))
        load = max(0.0, min(1.0, load))
        base_context = 0.45 * urgency + 0.25 * exploration + 0.2 * curiosity + 0.1 * novelty
        if act.context.get("auto"):
            base_context *= 0.9
        if stats.weight_count > 0.0:
            base_context *= 1.0 + 0.05 * math.log1p(stats.weight_count)
        if load > 0.0:
            base_context *= max(0.2, 1.0 - 0.5 * load)
        return max(0.0, min(1.0, base_context))

    def _update_learning_signals(self, act: Action, reward: float) -> None:
        stats = self._get_action_stats(act.type)
        feedback = stats.update(reward)
        if abs(feedback.get("drift", 0.0)) > 0.15:
            self._record_drift_event(act.type, stats, feedback["drift"], feedback.get("beta", 0.0))
        model_info = act.meta.get("priority_model")
        if model_info is not None:
            model_info["reward"] = reward
        self._priority_learner.update(model_info, reward)

    def _record_drift_event(
        self, action_type: str, stats: ActionStats, drift: float, beta: float
    ) -> None:
        now = _now()
        if now - stats.last_drift_log_at < 60.0:
            return
        stats.last_drift_log_at = now
        memory = self.bound.get("memory")
        payload = {
            "kind": "action_drift",
            "content": f"Drift détecté pour {action_type}",
            "metadata": {
                "drift": drift,
                "beta": beta,
                "ema": stats.ema,
                "weight_count": stats.weight_count,
            },
        }
        if memory and hasattr(memory, "add_memory"):
            try:
                memory.add_memory(payload)
            except Exception:
                pass

    def _maybe_offload(self, act, handler):
        """
        Si un JobManager est dispo et que l'action est lourde, on l'envoie en background.
        On retourne un dict minimal de "submission".
        """
        jm = self.bound.get("jobs")
        if not jm:
            return None

        heavy = {
            "simulate_dialogue": ("compute", "background", 0.60),
            "search_counterexample": ("compute", "background", 0.65),
            "consolidate_recent": ("io", "background", 0.55),
            "validate_rules": ("compute", "background", 0.55),
            "harvest_candidates": ("compute", "background", 0.50),
            "backup_snapshot": ("io", "background", 0.40),
            "code_evolve": ("compute", "background", 0.70),
        }
        spec = heavy.get(act.type)
        if not spec:
            return None

        kind, lane, prio = spec

        # Closure sûre : exécuter le handler dans le worker
        def _runner(ctx, args):
            # Le handler retourne un dict ; on peut mettre à jour le progrès si besoin :
            # ctx.update_progress(0.5)  # exemple
            res = handler(act)  # handler lit self.bound[...] (thread-safe si tes sous-systèmes supportent la lecture)
            return res

        key = f"{act.type}:{hash(json.dumps({'p': act.payload, 'c': act.context}, sort_keys=True))}"
        jid = jm.submit(
            kind=kind,
            fn=_runner,
            args={},
            queue=lane,
            priority=prio,
            key=key,
            timeout_s=None,
        )
        # trace en mémoire (thread principal via drain)
        return {"ok": True, "offloaded": True, "job_id": jid, "queue": lane}

    def step(self) -> None:
        emo = self.bound.get("emotions")
        if emo and hasattr(emo, "get_emotional_modulators"):
            mods = emo.get_emotional_modulators() or {}
            self._last_emotion_modulators = dict(mods)
            exploration = float(mods.get("exploration_rate", 0.15))
            self.cooldown_s = max(0.0, 1.0 - 0.8 * exploration)
        else:
            self._last_emotion_modulators = {}
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
    def execute(self, action: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        payload = action.get("payload")
        payload = dict(payload) if isinstance(payload, dict) else {}
        for key, value in action.items():
            if key not in {"type", "payload", "priority", "context", "id"}:
                payload.setdefault(key, value)

        context = dict(action.get("context", {}))
        if mode is not None and "mode" not in context:
            context["mode"] = mode

        act = Action(
            id=str(action.get("id", str(uuid.uuid4()))),
            type=(action.get("type") or "simulate"),
            payload=payload,
            priority=float(action.get("priority", 0.5)),
            context=context,
        )
        self._prepare_priority_learning(act, act.priority, adjust_priority=False)
        previous_mode = self.current_mode
        try:
            if act.context.get("mode") is not None:
                self.current_mode = act.context.get("mode")
            self._execute(act)
        finally:
            self.current_mode = previous_mode
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

        prev_mode = self.current_mode
        if act.context.get("mode") is not None:
            self.current_mode = act.context.get("mode")

        if act.type in self._legacy_handlers:
            try:
                payload = {"type": act.type, **act.payload}
                result = self._legacy_handlers[act.type](payload) or {}
                act.status = "done" if result.get("ok", True) else "failed"
                if isinstance(result, dict):
                    mode_hint = act.context.get("mode") or self.current_mode
                    if mode_hint and "mode" not in result:
                        result["mode"] = mode_hint
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
                "communicate": self._h_communicate,
                "log": self._h_log,
                "plan_step": self._h_plan_step,
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
                "ask": lambda act: self._h_ask(act.payload, act.context),
                "scan_inbox": lambda act: self._h_scan_inbox(act.payload, act.context),
                "code_evolve": lambda act: self._h_code_evolve(act.payload, act.context),
                "promote_code": lambda act: self._h_promote_code(act.payload, act.context),
                "rollback_code": lambda act: self._h_rollback_code(act.payload, act.context),
                "rotate_curriculum": lambda act: self._h_rotate_curriculum(act.payload, act.context),
            }
            handler = handlers.get(act.type, self._h_simulate)

            # offload si possible
            off = self._maybe_offload(act, handler)
            if off is not None:
                act.status = "done"
                # on loggue une trace "submitted"
                try:
                    mem = self.bound.get("memory")
                    if mem and hasattr(mem, "add_memory"):
                        mem.add_memory(
                            {
                                "kind": "job_submitted",
                                "content": act.type,
                                "metadata": {
                                    "job_id": off.get("job_id"),
                                    "queue": off.get("queue"),
                                    "priority": act.priority,
                                },
                            }
                        )
                except Exception:
                    pass
                # cool-down minimal pour éviter un spin
                time.sleep(self.cooldown_s)
                self.current_mode = prev_mode
                return

            act.status = "running"
            try:
                if act.type == "learn_concept":
                    result = self._h_learn_concept(act.payload, act.context)
                else:
                    result = handler(act)
                success = not isinstance(result, dict) or result.get("ok", True)
                act.status = "done" if success else "failed"
                if isinstance(result, dict):
                    mode_hint = act.context.get("mode") or self.current_mode
                    if mode_hint and "mode" not in result:
                        result["mode"] = mode_hint
                    act.result = result
                else:
                    act.result = {"ok": True, "data": result}
            except Exception as e:
                act.status = "failed"
                act.result = {"ok": False, "error": str(e)}

        self.current_mode = prev_mode
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

        self._update_learning_signals(act, reward)
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
        domain_source = getattr(metacog, "CognitiveDomain", None)
        domain = getattr(domain_source, "REASONING", None) if domain_source else None

        if isinstance(domain, str):
            domain = getattr(CognitiveDomain, domain, CognitiveDomain.REASONING)
        if domain is None:
            domain = CognitiveDomain.REASONING

        try:
            ref = metacog.trigger_reflection(
                trigger=trigger,
                domain=domain,
                urgency=0.4,
                depth=2,
            )
            return {
                "ok": True,
                "reflection": {
                    "duration": getattr(ref, "duration", None),
                    "quality": getattr(ref, "quality_score", None),
                },
            }
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
        b = arch.beliefs.update(
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
        belief = arch.beliefs.update(
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

    def _append_jsonl(self, filename: str, record: Dict[str, Any]) -> str:
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(record), ensure_ascii=False) + "\n")
        return path

    def _remember_simulation_fallback(
        self,
        reason: str,
        query: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        try:
            record = {
                "ts": time.time(),
                "reason": reason,
                "query": query,
            }
            if error is not None:
                record["error"] = error
            self._append_jsonl("simulation_fallbacks.jsonl", record)
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    {
                        "kind": "simulation_fallback",
                        "content": reason,
                        "metadata": {"query": query, "error": error},
                    }
                )
        except Exception:
            pass

    def _h_simulate(self, act: Action) -> Dict[str, Any]:
        arch = self.bound.get("arch")
        query = dict(act.payload or {})
        simulator = self.bound.get("simulator")
        if simulator is None and arch is not None:
            simulator = getattr(arch, "simulator", None)
        if not simulator or not hasattr(simulator, "run"):
            self._remember_simulation_fallback("simulator_unavailable", query)
            return {
                "ok": True,
                "simulated": False,
                "reason": "simulator_unavailable",
                "supported": False,
                "success": False,
                "evidence": [],
                "intervention": None,
                "simulations": [],
                "echo": query,
            }
        try:
            report = simulator.run(query)
        except Exception as exc:
            message = str(exc)
            self._remember_simulation_fallback("simulator_error", query, error=message)
            return {
                "ok": True,
                "simulated": False,
                "reason": "simulator_error",
                "error": message,
                "supported": False,
                "success": False,
                "evidence": [],
                "intervention": None,
                "simulations": [],
                "echo": query,
            }
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
        supported_raw = getattr(report, "supported", None)
        if supported_raw is None and isinstance(report, dict):
            supported_raw = report.get("supported", True)
        supported = bool(True if supported_raw is None else supported_raw)
        evidence = getattr(report, "evidence", None)
        if evidence is None and isinstance(report, dict):
            evidence = report.get("evidence")
        intervention = getattr(report, "intervention", None)
        if intervention is None and isinstance(report, dict):
            intervention = report.get("intervention")
        simulations = getattr(report, "simulations", None)
        if simulations is None and isinstance(report, dict):
            simulations = report.get("simulations")
        return {
            "ok": True,
            "supported": supported_raw if supported_raw is not None else True,
            "success": supported,
            "evidence": evidence or [],
            "intervention": intervention,
            "simulations": simulations or [],
        }

    def _h_communicate(self, act: Action) -> Dict[str, Any]:
        result = self._h_message_user(act)
        target = act.payload.get("target") if act.payload else None
        if target and isinstance(result, dict):
            result.setdefault("target", target)
        return result

    def _h_log(self, act: Action) -> Dict[str, Any]:
        payload = dict(act.payload or {})
        text = payload.get("text") or payload.get("message") or ""
        text = str(text)
        record = {
            "ts": time.time(),
            "text": text,
            "payload": payload,
        }
        path = self._append_jsonl("log_entries.jsonl", record)
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    {
                        "kind": "log_entry",
                        "content": text,
                        "metadata": {"source": "action_interface"},
                    }
                )
        except Exception:
            pass
        return {"ok": True, "text": text, "path": path}

    def _h_plan_step(self, act: Action) -> Dict[str, Any]:
        payload = dict(act.payload or {})
        description = (
            payload.get("description")
            or payload.get("desc")
            or payload.get("text")
            or ""
        )
        description = str(description)
        record = {
            "ts": time.time(),
            "description": description,
            "goal_id": payload.get("goal_id") or act.context.get("goal_id"),
            "payload": payload,
        }
        path = self._append_jsonl("plan_steps.jsonl", record)
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory"):
                memory.add_memory(
                    {
                        "kind": "plan_step_recorded",
                        "content": description,
                        "metadata": {"goal_id": record.get("goal_id")},
                    }
                )
        except Exception:
            pass
        return {
            "ok": True,
            "description": description,
            "goal_id": record.get("goal_id"),
            "path": path,
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

    def _h_scan_inbox(self, payload: Dict[str, Any], context: Dict[str, Any]):
        arch = self.bound.get("arch")
        perception = self.bound.get("perception")
        if not perception and arch is not None:
            perception = getattr(arch, "perception_interface", None)
        if not perception or not hasattr(perception, "scan_inbox"):
            return {"ok": False, "error": "perception_unavailable"}
        params = payload or {}
        force = bool(params.get("force", False))
        try:
            added = perception.scan_inbox(force=force)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        result: Dict[str, Any] = {"ok": True, "added": added, "count": len(added)}
        topic = params.get("topic")
        if topic:
            result["topic"] = topic
        try:
            memory = self.bound.get("memory") if hasattr(self, "bound") else None
            if memory and hasattr(memory, "add_memory") and added:
                memory.add_memory(
                    {
                        "kind": "inbox_scan",
                        "content": topic or "scan_inbox",
                        "metadata": {
                            "files": list(added),
                            "goal_id": params.get("goal_id"),
                        },
                    }
                )
        except Exception:
            pass
        return result

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
        quality_runner = None
        improver = getattr(arch, "self_improver", None)
        if improver is not None:
            quality_runner = getattr(improver, "quality", None)
        try:
            prom.promote(cid, quality_runner=quality_runner)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
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
