"""Background maintenance scheduler for the cognitive architecture.

The previous version of this module had two issues that prevented the
scheduler from running correctly once imported:

* The standard library modules it relied on (``time``, ``os``, ``json`` …)
  were never imported.  Python therefore raised ``NameError`` exceptions as
  soon as the helper functions such as :func:`_now` or :func:`_safe_json`
  were executed.
* The reflection task attempted to import ``CognitiveDomain`` from a
  top-level ``metacognition`` module.  Inside the package hierarchy the
  correct import path is ``AGI_Evolutive.metacognition``; the truncated
  import made the task crash at runtime.

Both problems manifested as "truncated command"/runtime failures even
though the file passed static syntax checks.  Restoring the missing imports
and using the fully qualified package path fixes the scheduler logic.
"""

import json
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, List

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.core.global_workspace import GlobalWorkspace
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore
from AGI_Evolutive.cognition.principle_inducer import PrincipleInducer


def _now() -> float:
    return time.time()


def _safe_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(obj), f, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_sanitize(obj), ensure_ascii=False) + "\n")
class Scheduler:
    """
    Orchestrateur de cycles :
      - tâches déclaratives avec intervalle (s)
      - exécution en thread daemon
      - reprise après crash (persist last_run)
      - instrumentation JSONL
    Par défaut, enchaîne :
      homeostasis → consolidation → concept_extractor → episodic_linker →
      goals/planning → reflection → evolution_manager

    À ne pas confondre avec :class:`AGI_Evolutive.light_scheduler.LightScheduler`
    (planificateur léger synchronisé à appeler dans une boucle).  Cette
    version runtime persiste son état et tourne dans un thread dédié.
    """

    def __init__(self, arch, data_dir: str = "data"):
        self.arch = arch
        self.data_dir = data_dir
        self.paths = {
            "state": os.path.join(self.data_dir, "runtime/scheduler_state.json"),
            "log": os.path.join(self.data_dir, "runtime/scheduler.log.jsonl"),
        }
        os.makedirs(os.path.dirname(self.paths["state"]), exist_ok=True)
        self.state = _safe_json(self.paths["state"], {
            "created_at": _now(),
            "last_runs": {},   # task_name -> ts
            "enabled": True
        })
        self.running = False
        self.thread = None

        # registre des tâches (nom -> dict)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._register_default_tasks()

        self.workspace = getattr(self.arch, "global_workspace", None)

        policy = getattr(self.arch, "policy", None)
        mechanism_store = None
        if policy is not None and hasattr(policy, "_mechanisms"):
            mechanism_store = getattr(policy, "_mechanisms", None)
        if mechanism_store is None:
            mechanism_store = getattr(self.arch, "mechanism_store", None)
        if mechanism_store is None:
            mechanism_store = MechanismStore()
            if policy is not None and hasattr(policy, "_mechanisms"):
                try:
                    policy._mechanisms = mechanism_store
                except Exception:
                    pass
        setattr(self.arch, "mechanism_store", mechanism_store)
        self.mechanism_store = mechanism_store

        principle_inducer = getattr(self.arch, "principle_inducer", None)
        if principle_inducer is None:
            principle_inducer = PrincipleInducer(self.mechanism_store)
            setattr(self.arch, "principle_inducer", principle_inducer)
        self.principle_inducer = principle_inducer

        period = getattr(self.arch, "evolution_period", None)
        if period is None:
            period = getattr(self, "_evolution_period", None)
        if period is None:
            period = 20
        try:
            period_value = int(period)
        except Exception:
            try:
                period_value = int(float(period))
            except Exception:
                period_value = 20
        self._evolution_period = max(1, period_value)
        try:
            self._tick = int(getattr(self, "_tick", 0))
        except Exception:
            self._tick = 0
        try:
            self._tick_counter = int(getattr(self, "_tick_counter", self._tick))
        except Exception:
            self._tick_counter = self._tick
        self._last_applicable_mais: List[Any] = []

    # ---------- helpers ----------
    def _build_state_snapshot(self) -> Dict[str, Any]:
        arch = self.arch
        language = getattr(arch, "language", None)
        dialogue = None
        if language is not None:
            dialogue = getattr(language, "state", None)
            dialogue = getattr(language, "dialogue_state", dialogue)
        world = getattr(arch, "world_model", None)
        return {
            "beliefs": getattr(arch, "beliefs", None),
            "self_model": getattr(arch, "self_model", None),
            "dialogue": dialogue,
            "world": world,
            "memory": getattr(arch, "memory", None),
        }

    def _predicate_registry(self, state: Dict[str, Any]) -> Dict[str, Callable[..., bool]]:
        policy = getattr(self.arch, "policy", None)
        if policy is not None and hasattr(policy, "build_predicate_registry"):
            try:
                registry = policy.build_predicate_registry(state)
                if isinstance(registry, dict):
                    return registry
            except Exception:
                pass

        dialogue = state.get("dialogue")
        world = state.get("world")
        self_model = state.get("self_model")
        beliefs = state.get("beliefs")

        def _belief_contains(topic: Any) -> bool:
            if beliefs is None:
                return False
            for accessor in ("contains", "has_fact", "has_edge"):
                fn = getattr(beliefs, accessor, None)
                if callable(fn):
                    try:
                        if fn(topic):
                            return True
                    except Exception:
                        continue
            return False

        def _belief_confidence(topic: Any, threshold: float) -> bool:
            if beliefs is None:
                return False
            confidence_for = getattr(beliefs, "confidence_for", None)
            if not callable(confidence_for):
                return False
            try:
                return float(confidence_for(topic)) >= float(threshold)
            except Exception:
                return False

        registry: Dict[str, Callable[..., bool]] = {
            "request_is_sensitive": lambda st: getattr(dialogue, "is_sensitive", False) if dialogue else False,
            "audience_is_not_owner": lambda st: (
                getattr(dialogue, "audience_id", None) != getattr(dialogue, "owner_id", None)
                if dialogue
                else False
            ),
            "has_consent": lambda st: getattr(dialogue, "has_consent", False) if dialogue else False,
            "imminent_harm_detected": lambda st: getattr(world, "imminent_harm", False) if world else False,
            "has_commitment": lambda st, key: (
                self_model.has_commitment(key)
                if hasattr(self_model, "has_commitment")
                else False
            ),
            "belief_mentions": lambda st, topic: _belief_contains(topic),
            "belief_confidence_above": lambda st, topic, threshold: _belief_confidence(topic, threshold),
        }
        return registry

    def _get_workspace(self) -> Any:
        workspace = getattr(self, "workspace", None)
        if workspace is None:
            workspace = getattr(self.arch, "global_workspace", None)
        if workspace is None:
            policy = getattr(self.arch, "policy", None)
            if policy is None:
                return None
            workspace = GlobalWorkspace(policy=policy, planner=getattr(self.arch, "planner", None))
            setattr(self.arch, "global_workspace", workspace)
        self.workspace = workspace
        return workspace

    def _render_and_emit(self, decision: Dict[str, Any], state: Dict[str, Any]) -> None:
        arch = self.arch
        language = getattr(arch, "language", None)
        text = decision.get("decision_text") or decision.get("decision") or "noop"
        if language is not None and hasattr(language, "renderer"):
            renderer = getattr(language, "renderer", None)
            if renderer and hasattr(renderer, "render_reply"):
                try:
                    semantics = {"text": decision.get("decision_text") or decision.get("decision", ""), "meta": decision}
                    ctx = {
                        "last_message": getattr(getattr(language, "state", None), "last_user", ""),
                        "omitted_content": decision.get("omitted_content", False),
                        "state_snapshot": state,
                    }
                    text = renderer.render_reply(semantics, ctx)
                except Exception:
                    text = decision.get("decision_text") or text
        arch.last_output_text = text
        logger = getattr(arch, "logger", None)
        if logger and hasattr(logger, "write"):
            try:
                logger.write("gw.decision", decision=decision, rendered=text)
            except Exception:
                pass

    # ---------- registration ----------
    def register(self, name: str, fn: Callable[[], None], interval_s: float, jitter_s: float = 0.0):
        self.tasks[name] = {
            "fn": fn, "interval": float(interval_s), "jitter": float(jitter_s)
        }

    def _register_default_tasks(self):
        # Les fonctions sont toutes robustifiées (attr checks)
        self.register("homeostasis", self._task_homeostasis, interval_s=60.0, jitter_s=5.0)
        self.register("consolidation", self._task_consolidation, interval_s=180.0, jitter_s=10.0)
        self.register("concepts", self._task_concepts, interval_s=45.0, jitter_s=5.0)
        self.register("episodes", self._task_episodes, interval_s=60.0, jitter_s=5.0)
        self.register("planning", self._task_planning, interval_s=90.0, jitter_s=8.0)
        self.register("reflection", self._task_reflection, interval_s=120.0, jitter_s=10.0)
        self.register("evolution", self._task_evolution, interval_s=120.0, jitter_s=10.0)

    # ---------- run loop ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _loop(self):
        while self.running and self.state.get("enabled", True):
            now = _now()
            for name, cfg in self.tasks.items():
                last = float(self.state["last_runs"].get(name, 0.0))
                due = last + cfg["interval"]
                if now >= due:
                    # jitter léger
                    j = cfg["jitter"]
                    if j > 0:
                        time.sleep(min(j, 0.25))  # lissé

                    t0 = _now()
                    ok = True
                    err = None
                    try:
                        cfg["fn"]()
                    except Exception as e:
                        ok = False
                        err = f"{e}\n{traceback.format_exc()}"
                    t1 = _now()

                    self.state["last_runs"][name] = t1
                    _write_json(self.paths["state"], self.state)
                    _append_jsonl(self.paths["log"], {
                        "t0": t0, "t1": t1, "dt": t1 - t0, "task": name, "ok": ok, "err": err
                    })
            time.sleep(0.5)

    # ---------- individual tasks (robustes) ----------
    def _task_homeostasis(self):
        # module emotions/homeostasis (si existant)
        hm = getattr(self.arch, "homeostasis", None)
        if hm and hasattr(hm, "run_homeostasis_cycle"):
            hm.run_homeostasis_cycle()
        # sinon, essayer emotions.adjust_if_needed()
        emo = getattr(self.arch, "emotions", None)
        if emo and hasattr(emo, "adjust_if_needed"):
            try:
                emo.adjust_if_needed()
            except Exception:
                pass

    def _task_consolidation(self):
        # mémoire/learning : consolidation (si dispo)
        # Essais : learning.consolidate(), memory.consolidate(), memory.consolidator.run_once_now()
        learn = getattr(self.arch, "learning", None)
        mem = getattr(self.arch, "memory", None)
        # learning
        if learn and hasattr(learn, "consolidate"):
            try:
                learn.consolidate()
                return
            except Exception:
                pass
        # memory
        if mem and hasattr(mem, "consolidate"):
            try:
                mem.consolidate()
                return
            except Exception:
                pass
        # consolidator (style VIKI+)
        cons = getattr(self.arch, "consolidator", None)
        if cons and hasattr(cons, "run_once_now"):
            try:
                cons.run_once_now(scope="auto")
            except Exception:
                pass

    def _task_concepts(self):
        ce = getattr(self.arch, "concept_extractor", None)
        if ce and hasattr(ce, "step"):
            ce.step()

    def _task_episodes(self):
        el = getattr(self.arch, "episodic_linker", None)
        if el and hasattr(el, "step"):
            el.step()
        workspace = self._get_workspace()
        state = self._build_state_snapshot()
        predicate_registry = self._predicate_registry(state)

        applicable: List[Any] = []
        try:
            applicable = list(self.mechanism_store.scan_applicable(state, predicate_registry))
        except Exception:
            applicable = []
        self._last_applicable_mais = applicable

        if workspace is None:
            return

        for mechanism in applicable:
            try:
                bids = list(mechanism.propose(state))
            except Exception:
                continue
            for bid in bids:
                try:
                    workspace.submit(bid)
                except AttributeError:
                    attention = max(0.0, min(1.0, getattr(bid, "expected_info_gain", 0.0)))
                    workspace.submit_bid(
                        bid.payload.get("origin", getattr(bid, "source", "mai")),
                        bid.action_hint,
                        attention,
                        bid.payload,
                    )

    def _task_planning(self):
        goals = getattr(self.arch, "goals", None)
        # Ex : goals.refresh_plans(), goals.step(), planner.plan_for_goal() …
        if goals and hasattr(goals, "step"):
            try:
                goals.step()
                return
            except Exception:
                pass
        if goals and hasattr(goals, "refresh_plans"):
            try:
                goals.refresh_plans()
            except Exception:
                pass
        workspace = self._get_workspace()
        policy = getattr(self.arch, "policy", None)
        if workspace and policy and hasattr(workspace, "step") and hasattr(policy, "decide_with_bids"):
            state = self._build_state_snapshot()
            try:
                workspace.step(state, timebox_iters=2)
                winners = list(workspace.winners())
            except Exception:
                winners = []
            if not winners:
                try:
                    winners = list(workspace.last_trace())
                except Exception:
                    winners = []
            try:
                decision = policy.decide_with_bids(
                    winners,
                    state,
                    global_workspace=workspace,
                    proposer=getattr(self.arch, "proposer", None),
                    homeo=getattr(self.arch, "homeostasis", None) or getattr(self.arch, "emotions", None),
                    planner=getattr(self.arch, "planner", None),
                    ctx={"scheduler": True, "workspace_trace": [bid.origin_tag() for bid in winners]},
                )
            except Exception:
                decision = None

            if decision:
                emitted = False
                runtime = getattr(self.arch, "runtime", None)
                response_api = getattr(runtime, "response", None) if runtime else None
                if (
                    runtime
                    and hasattr(runtime, "emit")
                    and response_api is not None
                    and hasattr(response_api, "format_agent_reply")
                ):
                    try:
                        utterance = response_api.format_agent_reply(decision)
                        runtime.emit(utterance)
                        emitted = True
                    except Exception:
                        emitted = False
                if not emitted:
                    self._render_and_emit(decision, state)

                applicable_mais = list(getattr(self, "_last_applicable_mais", []))
                if applicable_mais:
                    try:
                        from AGI_Evolutive.social.social_critic import SocialCritic

                        critic = SocialCritic()
                        outcome = {}
                        if hasattr(critic, "last_outcome"):
                            try:
                                outcome = critic.last_outcome() or {}
                            except Exception:
                                outcome = {}
                        for mechanism in applicable_mais:
                            try:
                                wins = 1.0 if any(getattr(bid, "source", "") == f"MAI:{mechanism.id}" for bid in winners) else 0.0
                                feedback = {
                                    "activation": 1.0,
                                    "wins": wins,
                                    "benefit": float(outcome.get("trust_delta", 0.0))
                                    - float(outcome.get("harm_delta", 0.0)),
                                    "regret": float(outcome.get("regret", 0.0)),
                                }
                                if hasattr(mechanism, "update_from_feedback"):
                                    mechanism.update_from_feedback(feedback)
                                try:
                                    self.mechanism_store.update(mechanism)
                                except Exception:
                                    pass
                            except Exception:
                                continue
                    except Exception:
                        pass

    def _task_reflection(self):
        mc = getattr(self.arch, "metacognition", None) or getattr(self.arch, "metacognitive_system", None)
        if mc and hasattr(mc, "trigger_reflection"):
            try:
                # réflexion légère récurrente (domain REASONING par défaut)
                from AGI_Evolutive.metacognition import CognitiveDomain
                mc.trigger_reflection(
                    trigger="periodic_scheduler",
                    domain=CognitiveDomain.REASONING,
                    urgency=0.3,
                    depth=1
                )
            except Exception:
                pass

    def _task_evolution(self):
        evo = getattr(self.arch, "evolution", None)
        if evo and hasattr(evo, "record_cycle"):
            try:
                # enregistre snapshot + génère recommandations
                evo.record_cycle(extra_tags={"via": "scheduler"})
                # potentiellement proposer des évolutions (non destructif)
                evo.propose_evolution()
            except Exception:
                pass
        self._tick = getattr(self, "_tick", 0) + 1
        self._tick_counter = self._tick
        if self._tick % self._evolution_period == 0:
            arch = self.arch
            recent_docs: List[Any]
            recent_dialogues: List[Any]

            if hasattr(arch, "recent_docs"):
                recent_docs = list(getattr(arch, "recent_docs") or [])
            else:
                recent_docs = []
                memory = getattr(arch, "memory", None)
                if memory and hasattr(memory, "get_recent_memories"):
                    try:
                        recent_docs = memory.get_recent_memories(n=200)
                    except Exception:
                        recent_docs = []

            if hasattr(arch, "recent_dialogues"):
                recent_dialogues = list(getattr(arch, "recent_dialogues") or [])
            else:
                recent_dialogues = []
                memory = getattr(arch, "memory", None)
                dialogue_log = getattr(memory, "interactions", None)
                if dialogue_log and hasattr(dialogue_log, "get_recent"):
                    try:
                        recent_dialogues = dialogue_log.get_recent(100)
                    except Exception:
                        recent_dialogues = []

            metrics_snapshot: Dict[str, Any] = {}
            metrics = getattr(arch, "metrics", None)
            if metrics and hasattr(metrics, "snapshot"):
                try:
                    metrics_snapshot = metrics.snapshot() or {}
                except Exception:
                    metrics_snapshot = {}

            try:
                self.principle_inducer.run(recent_docs, recent_dialogues, metrics_snapshot)
            except Exception:
                pass
