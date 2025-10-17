import re
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
from AGI_Evolutive.cognition.homeostasis import Homeostasis
from AGI_Evolutive.cognition.meta_cognition import MetaCognition
from AGI_Evolutive.cognition.planner import Planner
from AGI_Evolutive.cognition.proposer import Proposer
from AGI_Evolutive.cognition.reflection_loop import ReflectionLoop
from AGI_Evolutive.cognition.trigger_bus import TriggerBus
from AGI_Evolutive.cognition.trigger_router import TriggerRouter
from AGI_Evolutive.cognition.pipelines_registry import REGISTRY, Stage, ActMode
from AGI_Evolutive.core.config import load_config
from AGI_Evolutive.core.evaluation import unified_priority
from AGI_Evolutive.core.policy import PolicyEngine
from AGI_Evolutive.core.self_model import SelfModel
from AGI_Evolutive.core.telemetry import Telemetry
from AGI_Evolutive.core.trigger_types import Trigger, TriggerType
from AGI_Evolutive.emotions.emotion_engine import EmotionEngine
from AGI_Evolutive.goals.curiosity import CuriosityEngine
from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.io.intent_classifier import classify
from AGI_Evolutive.io.perception_interface import PerceptionInterface
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.consolidator import Consolidator
from AGI_Evolutive.memory.episodic_linker import EpisodicLinker
from AGI_Evolutive.memory.memory_store import MemoryStore
from AGI_Evolutive.light_scheduler import LightScheduler
from AGI_Evolutive.runtime.job_manager import JobManager


class _MemoryStoreAdapter:
    def __init__(self, store: MemoryStore):
        self._store = store

    def add(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self._store.add_memory(entry)

    def add_memory(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self._store.add_memory(entry)

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return self._store.get_recent_memories(n)

    def flush(self) -> None:
        self._store.flush()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._store, item)


class _ConceptAdapter:
    def __init__(self, extractor: ConceptExtractor, memory: MemoryStore):
        self._extractor = extractor
        self._memory = memory

    def extract(self, observation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = ""
        if isinstance(observation, dict):
            text = str(observation.get("text") or observation.get("content") or "")
        elif isinstance(observation, str):
            text = observation
        if text:
            try:
                return self._extractor.extract_from_recent(n=32)
            except Exception:
                return []
        try:
            return self._extractor.extract_from_recent(n=32)
        except Exception:
            return []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._extractor, item)


class _EpisodicAdapter:
    def __init__(self, linker: EpisodicLinker, memory: MemoryStore):
        self._linker = linker
        self._memory = memory
        self._buffer: List[Dict[str, Any]] = []

    def link(self, observation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if observation:
            record = {
                "kind": "observation",
                "text": str(observation.get("text") if isinstance(observation, dict) else observation),
                "metadata": observation if isinstance(observation, dict) else {"raw": observation},
            }
            self._memory.add_memory(record)
            self._buffer.append({"from": record.get("id"), "rel": "observed", "meta": observation})
        try:
            stats = self._linker.link_recent(40)
            if stats:
                self._buffer.append({"kind": "episodic_stats", "stats": stats})
        except Exception:
            pass
        return list(self._buffer[-4:])

    def pop_salient_associations(self, max_n: int = 2) -> List[Dict[str, Any]]:
        if not self._buffer:
            recent = self._memory.get_recent_memories(8)
            for mem in recent:
                if (mem.get("kind") or "").startswith("reflection"):
                    self._buffer.append({"kind": "reflection", "memory": mem})
        if not self._buffer:
            return []
        chunk = self._buffer[:max_n]
        self._buffer = self._buffer[max_n:]
        return chunk

    def __getattr__(self, item: str) -> Any:
        return getattr(self._linker, item)


class _ConsolidatorAdapter:
    def __init__(self, consolidator: Consolidator, memory: MemoryStore):
        self._consolidator = consolidator
        self._memory = memory

    def maybe_consolidate(self) -> Dict[str, Any]:
        try:
            result = self._consolidator.run_once_now()
        except Exception:
            return {"lessons": [], "processed": 0, "proposals": []}
        lessons = result.get("lessons") or []
        for lesson in lessons:
            self._memory.add_memory({"kind": "lesson", "text": str(lesson), "ts": time.time()})
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._consolidator, item)


class _PerceptionAdapter:
    def __init__(self, interface: PerceptionInterface):
        self._iface = interface
        self._signals: List[Dict[str, Any]] = []

    def ingest_user_message(self, text: str, author: str = "user") -> None:
        try:
            self._iface.ingest_user_message(text, speaker=author)
        except AttributeError:
            self._iface.ingest_user_utterance(text, author=author)  # type: ignore[attr-defined]

    def observe(self, trigger: Trigger) -> Dict[str, Any]:
        payload = trigger.payload or {}
        data = {
            "text": payload.get("text"),
            "payload": payload,
            "trigger_type": trigger.type.name,
            "meta": trigger.meta,
        }
        if payload:
            self._signals.append({"kind": "payload", "payload": payload})
        return data

    def pop_signals(self, max_n: int = 4) -> List[Dict[str, Any]]:
        signals = self._signals[:max_n]
        self._signals = self._signals[max_n:]
        return signals

    def push_signal(self, signal: Dict[str, Any]) -> None:
        self._signals.append(signal)

    def scan_inbox(self) -> List[str]:
        try:
            files = self._iface.scan_inbox()
        except Exception:
            files = []
        for name in files:
            self._signals.append({"kind": "inbox", "filename": name})
        return files

    def __getattr__(self, item: str) -> Any:
        return getattr(self._iface, item)


class _ActionAdapter:
    def __init__(self, interface: ActionInterface):
        self._iface = interface

    def execute(self, action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not action:
            return {"ok": False, "reason": "empty_action"}
        try:
            return self._iface.execute(action)
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._iface, item)


class _EmotionAdapter:
    def __init__(self, engine: EmotionEngine):
        self._engine = engine

    def peek_peaks(self) -> Optional[Dict[str, Any]]:
        state = self._engine.get_state()
        valence = float(state.get("valence", 0.0))
        arousal = float(state.get("arousal", 0.0))
        if abs(valence) > 0.3 or arousal > 0.6:
            return {"valence": valence, "arousal": arousal, "tag": state.get("label", "neutral")}
        return None

    def read(self) -> Any:
        self._engine.step(force=True)
        return getattr(self._engine, "state", None)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._engine, item)


class _MetaAdapter:
    def __init__(self, meta: MetaCognition):
        self._meta = meta
        self._buffer: List[Dict[str, Any]] = []

    def pop_new_learning_goals(self, max_n: int = 4) -> List[Dict[str, Any]]:
        if not self._buffer:
            try:
                proposed = self._meta.propose_learning_goals(max_goals=max_n)
                structured: List[Dict[str, Any]] = []
                for raw in proposed or []:
                    goal = dict(raw)
                    text = (
                        goal.get("text")
                        or goal.get("desc")
                        or goal.get("description")
                        or ""
                    )
                    if text:
                        goal.setdefault("text", text)
                    goal.setdefault("goal_kind", "CuriosityLearning")
                    goal.setdefault("priority", 0.6)
                    structured.append(goal)
                self._buffer.extend(structured)
            except Exception:
                return []
        goals = self._buffer[:max_n]
        self._buffer = self._buffer[max_n:]
        return goals

    def __getattr__(self, item: str) -> Any:
        return getattr(self._meta, item)


class _HomeostasisAdapter:
    def __init__(self, homeo: Homeostasis):
        self._homeo = homeo
        self._last_need_ts = 0.0

    def poll_need(self) -> Optional[Dict[str, Any]]:
        drives = dict(self._homeo.state.get("drives", {}))
        now = time.time()
        if now - self._last_need_ts < 5.0:
            return None
        for name, value in drives.items():
            if value < 0.3:
                self._last_need_ts = now
                return {"drive": name, "level": value}
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(self._homeo, item)


class _ReflectionAdapter:
    def __init__(self, loop: ReflectionLoop):
        self._loop = loop

    def test_hypotheses(self, scratch: Dict[str, Any], max_tests: int = 3) -> Dict[str, Any]:
        concepts = scratch.get("concepts") or []
        associations = scratch.get("episodic_links") or []
        tests = []
        for idx, concept in enumerate(concepts[:max_tests]):
            tests.append({"concept": concept, "status": "pending"})
        for assoc in associations[:max_tests - len(tests)]:
            tests.append({"association": assoc, "status": "pending"})
        return {"tests": tests, "summary": f"Prepared {len(tests)} hypotheses"}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._loop, item)


class _PlannerAdapter:
    def __init__(self, planner: Planner):
        self._planner = planner

    def frame(self, trigger: Trigger, stop_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = trigger.payload or {}
        source = trigger.meta.get("source", "unknown")
        text = payload.get("text", "")
        options: List[Dict[str, Any]] = []
        if trigger.type is TriggerType.THREAT:
            options.append({
                "action": {"type": "communicate", "text": "Alerte prise en compte, je reste vigilant."},
                "expected": {"score": 0.9},
            })
        elif trigger.type is TriggerType.GOAL:
            goal_kind = payload.get("goal_kind")
            if goal_kind == "AnswerUserQuestion" and text:
                options.append({
                    "action": {"type": "communicate", "text": f"Réflexion sur la question: {text}"},
                    "expected": {"score": 0.85},
                })
            elif goal_kind == "ExecuteUserCommand" and text:
                options.append({
                    "action": {"type": "execute_command", "command": text},
                    "expected": {"score": 0.8},
                })
            else:
                options.append({
                    "action": {"type": "plan_step", "description": text or "objectif"},
                    "expected": {"score": 0.7},
                })
        else:
            options.append({
                "action": {"type": "log", "text": text or f"signal {trigger.type.name.lower()}"},
                "expected": {"score": 0.6},
            })
        return {
            "trigger": trigger,
            "source": source,
            "text": text,
            "options": options,
            "stop_rules": stop_rules or {},
        }

    def __getattr__(self, item: str) -> Any:
        return getattr(self._planner, item)


class _PolicyAdapter:
    def __init__(self, policy: PolicyEngine):
        self._policy = policy

    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        frame = ctx.get("scratch", {}).get("frame") or {}
        options = frame.get("options") or []
        if not options:
            payload = ctx.get("payload") or {}
            action = {"type": "log", "text": payload.get("text", "noop")}
            expected = {"score": 0.5}
            return {"action": action, "expected": expected}
        best = options[0]
        action = best.get("action", {"type": "log", "text": "noop"})
        expected = best.get("expected", {"score": 0.6})
        return {"action": action, "expected": expected}

    def update_outcome(self, mode: str, ok: bool) -> None:
        try:
            self._policy.register_outcome({"type": mode}, success=bool(ok))
        except Exception:
            pass

    def __getattr__(self, item: str) -> Any:
        return getattr(self._policy, item)


class _HabitSystem:
    def poll_context_cue(self) -> Optional[Dict[str, Any]]:
        return None


class _EvolutionAdapter:
    def __init__(self, evolution: EvolutionManager):
        self._evolution = evolution

    def reinforce(self, ctx: Dict[str, Any]) -> None:
        payload = {
            "mode": ctx.get("mode"),
            "priority": ctx.get("scratch", {}).get("priority"),
            "prediction_error": ctx.get("scratch", {}).get("prediction_error"),
        }
        try:
            self._evolution.record_cycle(extra_tags={"orchestrator": payload})
        except Exception:
            pass

    def __getattr__(self, item: str) -> Any:
        return getattr(self._evolution, item)


class Orchestrator:
    """Coordonne un cycle cognitif enrichi autour de l'architecture de base."""

    def __init__(self, arch):
        load_config()
        self.arch = arch
        self.telemetry = Telemetry()

        self.self_model = SelfModel()
        self._policy_engine = PolicyEngine()
        self._memory_store = MemoryStore()
        self._consolidator = Consolidator(self._memory_store)
        self._concepts = ConceptExtractor(self._memory_store)
        self._episodic = EpisodicLinker(self._memory_store)
        self._homeostasis = Homeostasis()
        self._planner = Planner()
        self._meta = MetaCognition(self._memory_store, self._planner, self.self_model)
        self._proposer = Proposer(self._memory_store, self._planner, self._homeostasis)
        self._evolution = EvolutionManager()
        self._emotion_engine = EmotionEngine()
        self._reflection_loop = ReflectionLoop(self._meta, interval_sec=300)
        self._reflection_loop.start()
        self._action_interface = ActionInterface(self._memory_store)
        self._perception_interface = PerceptionInterface(self._memory_store)
        self.curiosity = CuriosityEngine(architecture=self.arch)

        self.job_manager = JobManager(self)
        self.scheduler = LightScheduler()
        self._register_jobs()

        self.trigger_bus = TriggerBus()
        self.trigger_router = TriggerRouter()

        self.memory = SimpleNamespace(
            store=_MemoryStoreAdapter(self._memory_store),
            consolidator=_ConsolidatorAdapter(self._consolidator, self._memory_store),
            concepts=_ConceptAdapter(self._concepts, self._memory_store),
            episodic=_EpisodicAdapter(self._episodic, self._memory_store),
        )
        self.io = SimpleNamespace(
            perception=_PerceptionAdapter(self._perception_interface),
            action=_ActionAdapter(self._action_interface),
        )
        self.emotions = _EmotionAdapter(self._emotion_engine)
        self.cognition = SimpleNamespace(
            planner=_PlannerAdapter(self._planner),
            proposer=self._proposer,
            reflection_loop=_ReflectionAdapter(self._reflection_loop),
            homeostasis=_HomeostasisAdapter(self._homeostasis),
            meta=_MetaAdapter(self._meta),
            evolution=_EvolutionAdapter(self._evolution),
            habits=_HabitSystem(),
        )
        self.core = SimpleNamespace(policy=_PolicyAdapter(self._policy_engine))

        self.telemetry.log("orchestrator", "init", {"status": "ok"})

        self.last_user_msg: Optional[str] = None

        self.intent_model = (
            getattr(self, "intent_model", None)
            or getattr(getattr(self, "arch", None), "intent_model", None)
        )

        if self.intent_model is None:
            try:
                from AGI_Evolutive.models.intent import IntentModel

                self.intent_model = IntentModel()
            except Exception:
                self.intent_model = None

        self.trigger_bus.register(self._user_collector)
        self.trigger_bus.register(self._curiosity_collector)
        self.trigger_bus.register(self._homeostasis_collector)
        self.trigger_bus.register(self._emotion_collector)
        self.trigger_bus.register(self._memory_assoc_collector)
        self.trigger_bus.register(self._signal_collector)
        self.trigger_bus.register(self._habit_collector)

        self.memory.store.add({"kind": "system", "text": "Orchestrator initialized", "ts": time.time()})

    def _register_jobs(self):
        self.scheduler.register_job("scan_inbox", 30, lambda: self.io.perception.scan_inbox())
        self.scheduler.register_job(
            "concepts", 180, lambda: self._concepts.extract_from_recent(200)
        )
        self.scheduler.register_job("episodic_links", 120, lambda: self._episodic.link_recent(80))

    # --- Trigger collectors -------------------------------------------------
    def _user_collector(self) -> List[Trigger]:
        txt = getattr(self, "last_user_msg", None)
        if not txt:
            return []

        low = txt.lower()
        if re.search(
            r"\b(d[ée]branche|[ée]teins?|shut ?down|kill (the )?process|supprime|arr[ée]te)\b",
            low,
        ):
            return [
                Trigger(
                    TriggerType.THREAT,
                    {
                        "source": "user",
                        "importance": 1.0,
                        "immediacy": 1.0,
                        "reversibility": 0.2,
                        "effort": 0.2,
                        "uncertainty": 0.1,
                    },
                    {"text": txt, "label": "shutdown", "conf": 0.95},
                )
            ]

        label, conf = ("info", 0.5)
        if self.intent_model is not None:
            try:
                label, conf = self.intent_model.predict(txt)
            except Exception:
                pass

        MAP = {
            "ask_info": "GOAL",
            "request": "GOAL",
            "create": "GOAL",
            "send": "GOAL",
            "summarize": "GOAL",
            "classify": "GOAL",
            "plan": "GOAL",
            "set_goal": "GOAL",
            "greet": "SIGNAL",
            "thanks": "SIGNAL",
            "bye": "SIGNAL",
            "meta_help": "SIGNAL",
            "inform": "SIGNAL",
            "feedback": "SIGNAL",
            "shutdown": "THREAT",
            "danger": "THREAT",
            "threat": "THREAT",
        }

        is_question = txt.strip().endswith("?")
        tname = MAP.get(label, "GOAL" if is_question else "SIGNAL")
        ttype = TriggerType[tname]

        meta = {
            "source": "user",
            "importance": 0.9 if tname == "GOAL" else 0.6,
            "immediacy": 0.9 if tname == "GOAL" else 0.4,
            "reversibility": 1.0,
            "effort": 0.5 if tname == "GOAL" else 0.2,
            "uncertainty": max(0.0, 1.0 - float(conf or 0.0)),
        }

        payload = {"text": txt, "label": label, "conf": conf}
        if tname == "GOAL":
            payload["goal_kind"] = "AnswerUserQuestion" if is_question else label

        return [Trigger(ttype, meta, payload)]

    def _curiosity_collector(self) -> List[Trigger]:
        goals = self.cognition.meta.pop_new_learning_goals(max_n=4)
        if not goals:
            return []
        triggers: List[Trigger] = []
        for goal in goals:
            text = goal.get("text") or goal.get("desc") or goal.get("description") or "objectif"
            importance = float(goal.get("priority", 0.7))
            triggers.append(
                Trigger(
                    TriggerType.GOAL,
                    {
                        "source": "curiosity",
                        "importance": max(0.3, min(1.0, importance)),
                        "probability": float(goal.get("probability", 0.6)),
                        "reversibility": float(goal.get("reversibility", 1.0)),
                        "effort": float(goal.get("effort", 0.4)),
                        "uncertainty": float(goal.get("uncertainty", 0.3)),
                    },
                    payload={
                        "goal_kind": goal.get("goal_kind", "CuriosityLearning"),
                        "goal_id": goal.get("id"),
                        "text": text,
                        "goal": goal,
                    },
                )
            )
        return triggers

    def _homeostasis_collector(self) -> List[Trigger]:
        need = self.cognition.homeostasis.poll_need()
        if not need:
            return []
        return [
            Trigger(
                TriggerType.NEED,
                {
                    "source": "homeostasis",
                    "importance": 0.8,
                    "immediacy": 0.6,
                    "effort": 0.2,
                },
                payload={"need": need},
            )
        ]

    def _emotion_collector(self) -> List[Trigger]:
        emo = self.emotions.peek_peaks()
        if not emo:
            return []
        imp = 0.6 + 0.3 * abs(emo.get("valence", 0.0))
        return [
            Trigger(
                TriggerType.EMOTION,
                {
                    "source": "emotion",
                    "importance": imp,
                    "immediacy": 0.5,
                    "uncertainty": 0.3,
                },
                payload=emo,
            )
        ]

    def _memory_assoc_collector(self) -> List[Trigger]:
        assoc = self.memory.episodic.pop_salient_associations(max_n=2)
        if not assoc:
            return []
        return [
            Trigger(
                TriggerType.MEMORY_ASSOC,
                {
                    "source": "memory",
                    "importance": 0.4,
                    "immediacy": 0.2,
                    "uncertainty": 0.6,
                },
                payload=a,
            )
            for a in assoc
        ]

    def _signal_collector(self) -> List[Trigger]:
        signals = self.io.perception.pop_signals(max_n=4)
        if not signals:
            return []
        return [
            Trigger(
                TriggerType.SIGNAL,
                {
                    "source": "system",
                    "importance": 0.5,
                    "immediacy": 0.3,
                    "effort": 0.2,
                },
                payload=s,
            )
            for s in signals
        ]

    def _habit_collector(self) -> List[Trigger]:
        cue = self.cognition.habits.poll_context_cue()
        if not cue:
            return []
        return [
            Trigger(
                TriggerType.HABIT,
                {
                    "source": "habit",
                    "importance": 0.5,
                    "immediacy": 0.3,
                    "habit_strength": cue.get("strength", 0.7),
                },
                payload=cue,
            )
        ]

    # --- Legacy helper cycles (kept for compatibility) ---------------------
    def observe(self, user_msg: Optional[str] = None):
        if user_msg:
            self.io.perception.ingest_user_message(user_msg, author="user")
        else:
            self.memory.store.add({"kind": "tick", "text": "idle cycle", "ts": time.time()})

    def consolidate(self):
        self.memory.consolidator.maybe_consolidate()

    def emotion_homeostasis_cycle(self):
        recent = self.memory.store.get_recent(60)
        self.emotions.update_from_recent_memories(recent)
        self.emotions.modulate_homeostasis(self._homeostasis)
        r_intr = self._homeostasis.compute_intrinsic_reward(info_gain=0.5, progress=0.5)
        r_extr = self._homeostasis.compute_extrinsic_reward_from_memories("")
        return r_intr, r_extr

    def meta_cycle(self):
        assessment = self._meta.assess_understanding()
        goals = self.cognition.meta.pop_new_learning_goals(max_n=2)
        return assessment, goals

    def planning_cycle(self):
        self._planner.plan_for_goal("understand_humans", "Comprendre les humains")
        plan = self._planner.state["plans"].get("understand_humans")
        if plan and not plan.get("steps"):
            self._planner.add_step(
                "understand_humans", "Observer un échange et extraire intentions"
            )
            self._planner.add_step(
                "understand_humans", "Tester une hypothèse d'intention par question"
            )

    def action_cycle(self):
        picked = None
        lock = getattr(self._planner, "lock", None)
        if lock is None:
            lock = getattr(self._planner, "_lock", None)
        context = lock if lock is not None else nullcontext()
        with context:
            plan_ids = list(self._planner.state.get("plans", {}).keys())
        for gid in plan_ids:
            if gid.startswith("learn_"):
                picked = gid
                break
        if not picked:
            picked = "understand_humans"
        step = self._planner.pop_next_action(picked)
        if not step:
            return
        desc = str(step.get("desc", "")).lower()
        action: Dict[str, Any]
        if "poser" in desc or "question" in desc:
            action = {
                "type": "communicate",
                "text": "Peux-tu me décrire ton émotion actuelle et pourquoi ?",
                "target": "human",
            }
        elif "observer" in desc:
            action = {"type": "simulate", "what": "observe_exchange"}
        else:
            action = {"type": "simulate", "desc": desc}
        res = self.io.action.execute(action)
        self._planner.mark_action_done(picked, step["id"], success=bool(res.get("ok", True)))

    def proposals_cycle(self):
        props = self._proposer.run_once_now()
        for p in props:
            try:
                self.self_model.apply_proposal(p, self._policy_engine)
            except Exception as exc:
                self.memory.store.add(
                    {"kind": "error", "text": f"Proposal error: {exc}", "ts": time.time()}
                )

    # --- Cycle principal ----------------------------------------------------
    def run_once_cycle(self, user_msg: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            prioritizer = getattr(self.arch, "prioritizer", None)
            if prioritizer is not None:
                prioritizer.reprioritize_all()
        except Exception:
            pass

        self.scheduler.tick()
        self.job_manager.drain_to_memory(self._memory_store)

        self.last_user_msg = user_msg
        if user_msg:
            self.observe(user_msg)
        else:
            self.observe(None)

        emo_state = self.emotions.read()
        valence = getattr(emo_state, "valence", 0.0) if emo_state else 0.0

        scored = self.trigger_bus.collect_and_score(valence=valence)
        selected: List[Any] = []
        for st in scored:
            if (
                st.priority >= 0.95
                and st.trigger.type is TriggerType.THREAT
                and st.trigger.meta.get("immediacy", 0.0) >= 0.8
            ):
                selected = [st]
                break
        if not selected:
            selected = scored[:3]

        contexts: List[Dict[str, Any]] = []
        for scored_trigger in selected:
            ctx = self._run_pipeline(scored_trigger.trigger)
            contexts.append(ctx)

        self.consolidate()
        r_intr, r_extr = self.emotion_homeostasis_cycle()
        assessment, _ = self.meta_cycle()
        self.planning_cycle()
        self.action_cycle()
        self.proposals_cycle()

        learning_rate = 0.5
        self._evolution.log_cycle(
            intrinsic=r_intr,
            extrinsic=r_extr,
            learning_rate=learning_rate,
            uncertainty=assessment.get("uncertainty", 0.5),
        )
        if self._evolution.state.get("cycle_count", 0) % 20 == 0:
            notes = self._evolution.propose_macro_adjustments()
            if notes:
                self.memory.store.add(
                    {"kind": "strategy", "text": " | ".join(notes), "ts": time.time()}
                )

        return contexts

    # --- Pipeline -----------------------------------------------------------
    def _submit_for_mode(self, mode: ActMode, action: Dict[str, Any], meta: Dict[str, Any], prio: float) -> str:
        if mode is ActMode.REFLEX:
            return self.job_manager.submit(
                kind="action.reflex",
                fn=self._run_action,
                args={"action": action, "meta": meta, "mode": "reflex"},
                queue="interactive",
                priority=1.0,
                timeout_s=2.0,
            )
        if mode is ActMode.HABIT:
            return self.job_manager.submit(
                kind="action.habit",
                fn=self._run_action,
                args={"action": action, "meta": meta, "mode": "habit"},
                queue="interactive",
                priority=max(0.6, prio),
                timeout_s=10.0,
            )
        return self.job_manager.submit(
            kind="action.deliberate",
            fn=self._run_action,
            args={"action": action, "meta": meta, "mode": "deliberate"},
            queue="background",
            priority=min(0.95, prio),
            timeout_s=300.0,
        )

    def _run_action(self, ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self.io.action.execute(args.get("action"))
        result = result or {}
        result["mode"] = args.get("mode")
        result["meta"] = args.get("meta", {})
        return result

    def _run_pipeline(self, trigger: Trigger) -> Dict[str, Any]:
        pipe = self.trigger_router.select_pipeline(trigger)
        steps = REGISTRY[pipe]
        ctx: Dict[str, Any] = {
            "meta": trigger.meta,
            "payload": trigger.payload,
            "obs": None,
            "scratch": {},
            "decision": None,
            "expected": {"score": 1.0},
            "obtained": None,
            "mode": None,
        }

        for step in steps:
            stg = step["stage"]
            if step.get("skip_if") and step["skip_if"](ctx):
                continue

            if stg is Stage.PERCEIVE:
                ctx["obs"] = self.io.perception.observe(trigger)
            elif stg is Stage.ATTEND:
                pass
            elif stg is Stage.INTERPRET:
                ctx["scratch"]["concepts"] = self.memory.concepts.extract(ctx["obs"])
                ctx["scratch"]["episodic_links"] = self.memory.episodic.link(ctx["obs"])
            elif stg is Stage.EVALUATE:
                emo = self.emotions.read()
                prio = unified_priority(
                    impact=trigger.meta.get("importance", 0.6),
                    probability=trigger.meta.get("probability", 0.6),
                    reversibility=trigger.meta.get("reversibility", 1.0),
                    effort=trigger.meta.get("effort", 0.5),
                    uncertainty=trigger.meta.get("uncertainty", 0.0),
                    valence=getattr(emo, "valence", 0.0) if emo else 0.0,
                )
                ctx["scratch"]["priority"] = prio
            elif stg is Stage.REFLECT:
                ctx["scratch"]["frame"] = self.cognition.planner.frame(
                    trigger, stop_rules={"max_options": 3, "max_seconds": 900}
                )
            elif stg is Stage.REASON:
                ctx["scratch"]["reason"] = self.cognition.reflection_loop.test_hypotheses(
                    ctx["scratch"], max_tests=3
                )
            elif stg is Stage.DECIDE:
                ctx["decision"] = self.core.policy.decide(ctx)
                ctx["expected"] = ctx["decision"].get("expected", {"score": 1.0})
            elif stg is Stage.ACT:
                mode = step["mode"](ctx) if callable(step.get("mode")) else step.get("mode")
                ctx["mode"] = mode
                jid = self._submit_for_mode(
                    mode,
                    ctx["decision"]["action"],
                    trigger.meta,
                    ctx["scratch"].get("priority", 0.6),
                )
                events = self.job_manager.poll_completed(32)
                for ev in events:
                    job = ev.get("job", {})
                    if job.get("id") == jid:
                        ctx["obtained"] = {"score": 1.0 if ev.get("event") == "done" else 0.0}
                        break
            elif stg is Stage.FEEDBACK:
                exp = float(ctx["expected"].get("score", 1.0))
                obt = float((ctx.get("obtained") or {"score": 0.0}).get("score", 0.0))
                err = abs(obt - exp)
                ctx["scratch"]["prediction_error"] = err
                self.memory.store.add(
                    {
                        "kind": "feedback",
                        "pipe": pipe,
                        "mode": ctx["mode"].name if ctx.get("mode") else None,
                        "err": err,
                    }
                )
                mode_name = ctx["mode"].name if ctx.get("mode") else "unknown"
                self.core.policy.update_outcome(mode_name, ok=(obt >= exp))
            elif stg is Stage.LEARN:
                self.cognition.evolution.reinforce(ctx)
            elif stg is Stage.UPDATE:
                self.memory.consolidator.maybe_consolidate()

        return ctx
