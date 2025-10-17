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


EVOLUTION_PERIOD = 12


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
        self._tick_counter = 0

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
        gw = getattr(self.arch, "global_workspace", None)
        policy = getattr(self.arch, "policy", None)
        if gw is None and policy:
            gw = GlobalWorkspace(policy=policy, planner=getattr(self.arch, "planner", None))
            setattr(self.arch, "global_workspace", gw)
        mechanism_store = getattr(policy, "_mechanisms", None) if policy else None
        if gw and policy and mechanism_store and hasattr(policy, "build_predicate_registry"):
            try:
                state = self._build_state_snapshot()
                predicate_registry = policy.build_predicate_registry(state)
                for mai in mechanism_store.scan_applicable(state, predicate_registry):
                    for bid in mai.propose(state):
                        try:
                            gw.submit(bid)
                        except AttributeError:
                            gw.submit_bid(
                                bid.payload.get("origin", "mai"),
                                bid.action_hint,
                                max(0.0, min(1.0, bid.expected_info_gain)),
                                bid.payload,
                            )
            except Exception:
                pass

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
        gw = getattr(self.arch, "global_workspace", None)
        policy = getattr(self.arch, "policy", None)
        if gw is None and policy:
            gw = GlobalWorkspace(policy=policy, planner=getattr(self.arch, "planner", None))
            setattr(self.arch, "global_workspace", gw)
        if gw and policy and hasattr(gw, "step") and hasattr(policy, "decide_with_bids"):
            try:
                state = self._build_state_snapshot()
                gw.step(state, timebox_iters=2)
                winners = gw.winners()
            except Exception:
                winners = []
                state = self._build_state_snapshot()
            try:
                decision = policy.decide_with_bids(
                    winners,
                    state,
                    global_workspace=gw,
                    proposer=getattr(self.arch, "proposer", None),
                    homeo=getattr(self.arch, "homeostasis", None) or getattr(self.arch, "emotions", None),
                    planner=getattr(self.arch, "planner", None),
                    ctx={"scheduler": True},
                )
                self._render_and_emit(decision, state)
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
        self._tick_counter += 1
        if self._tick_counter % EVOLUTION_PERIOD == 0:
            interaction_miner = getattr(self.arch, "interaction_miner", None)
            principle_inducer = getattr(self.arch, "principle_inducer", None)
            metrics = getattr(self.arch, "metrics", None)
            memory = getattr(self.arch, "memory", None)
            recent_docs: List[Any] = []
            recent_dialogues: List[Any] = []
            if memory and hasattr(memory, "get_recent_memories"):
                try:
                    recent_docs = memory.get_recent_memories(n=200)
                except Exception:
                    recent_docs = []
            dialogue_log = getattr(memory, "interactions", None)
            if dialogue_log and hasattr(dialogue_log, "get_recent"):
                try:
                    recent_dialogues = dialogue_log.get_recent(100)
                except Exception:
                    recent_dialogues = []
            if interaction_miner and hasattr(interaction_miner, "extract_normative_patterns") and principle_inducer:
                try:
                    patterns = interaction_miner.extract_normative_patterns(recent_docs, recent_dialogues)
                    if hasattr(principle_inducer, "induce_from_patterns"):
                        candidates = principle_inducer.induce_from_patterns(patterns)
                        if hasattr(principle_inducer, "prefilter"):
                            snapshot = metrics.snapshot() if metrics and hasattr(metrics, "snapshot") else {}
                            candidates = principle_inducer.prefilter(candidates, history_stats=snapshot)
                        if hasattr(principle_inducer, "submit_for_evaluation"):
                            principle_inducer.submit_for_evaluation(candidates)
                except Exception:
                    pass
