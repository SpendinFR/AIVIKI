import json
import re
import time
from typing import Any, Dict, Optional, List, Tuple, Callable

from AGI_Evolutive.autonomy import AutonomyManager
from AGI_Evolutive.beliefs.graph import BeliefGraph, Evidence
from AGI_Evolutive.knowledge.ontology_facade import EntityLinker, Ontology
from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
from AGI_Evolutive.cognition.reward_engine import RewardEngine
from AGI_Evolutive.core.telemetry import Telemetry
from AGI_Evolutive.core.question_manager import QuestionManager
from AGI_Evolutive.creativity import CreativitySystem
from AGI_Evolutive.emotions import EmotionalSystem
from AGI_Evolutive.goals import GoalSystem
from AGI_Evolutive.goals.dag_store import GoalDAG
from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.io.perception_interface import PerceptionInterface
from AGI_Evolutive.language.understanding import SemanticUnderstanding
from AGI_Evolutive.language.style_policy import StylePolicy
from AGI_Evolutive.language.social_reward import extract_social_reward
from AGI_Evolutive.language.style_profiler import StyleProfiler
from AGI_Evolutive.language.nlg import NLGContext, apply_mai_bids_to_nlg
from AGI_Evolutive.learning import ExperientialLearning
from AGI_Evolutive.memory import MemorySystem
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.episodic_linker import EpisodicLinker
from AGI_Evolutive.memory.vector_store import VectorStore
from AGI_Evolutive.metacog.calibration import CalibrationMeter, NoveltyDetector
from AGI_Evolutive.metacognition import MetacognitiveSystem
from AGI_Evolutive.models import IntentModel, UserModel
from AGI_Evolutive.perception import PerceptionSystem
from AGI_Evolutive.reasoning import ReasoningSystem
from AGI_Evolutive.reasoning.abduction import AbductiveReasoner, Hypothesis
from AGI_Evolutive.reasoning.causal import CounterfactualSimulator, SCMStore
from AGI_Evolutive.reasoning.question_engine import QuestionEngine
from AGI_Evolutive.runtime.logger import JSONLLogger
from AGI_Evolutive.runtime.response import ensure_contract, format_agent_reply
from AGI_Evolutive.runtime.scheduler import Scheduler
from AGI_Evolutive.runtime.job_manager import JobManager
from AGI_Evolutive.world_model import PhysicsEngine
from AGI_Evolutive.self_improver import SelfImprover
from AGI_Evolutive.self_improver.code_evolver import CodeEvolver
from AGI_Evolutive.self_improver.promote import PromotionManager
from AGI_Evolutive.planning.htn import HTNPlanner
from AGI_Evolutive.core.persistence import PersistenceManager
from AGI_Evolutive.core.config import cfg


class CognitiveArchitecture:
    """Central coordinator for the agent's cognitive subsystems."""

    def __init__(self, boot_minimal: bool = False):
        self.boot_minimal = boot_minimal
        # Observability
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.telemetry = Telemetry()
        self.style_policy = StylePolicy()
        self.intent_model = IntentModel()
        self.question_manager = QuestionManager(self)
        self._last_intent_decay = time.time()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Global state
        self.global_activation = 0.5
        self.start_time = time.time()
        self.reflective_mode = True
        self.last_output_text = "OK"
        self.last_user_id = "default"

        # Core subsystems
        self.telemetry.log("init", "core", {"stage": "memory"})
        self.vector_store = VectorStore()
        self.memory = MemorySystem(self)
        from AGI_Evolutive.memory.semantic_manager import (  # type: ignore  # local import avoids circular init
            SemanticMemoryManager,
        )

        self.memory.semantic = SemanticMemoryManager(
            self.memory,
            architecture=self,
            index_backend=self.vector_store,
        )
        if getattr(self.memory, "retrieval", None) is not None:
            setattr(self.memory.retrieval, "vector_store", self.vector_store)

        # === RAG 5★ : init lazy & optionnel ===
        self.rag = None
        self.rag_cfg = None
        try:
            import json

            try:
                with open("configs/rag.json", "r", encoding="utf-8") as fh:
                    self.rag_cfg = json.load(fh)
            except Exception:
                self.rag_cfg = {
                    "retrieval": {
                        "topk_dense": 200,
                        "topk_sparse": 100,
                        "topk_fused": 80,
                        "alpha_dense": 0.6,
                        "beta_sparse": 0.4,
                        "recency_boost": 0.2,
                        "recency_half_life_days": 14.0,
                    },
                    "ann": {
                        "backend": "faiss",
                        "hnsw": True,
                        "efSearch": 128,
                        "M": 32,
                        "metric": "ip",
                    },
                    "rerank": {"topk": 30, "mmr_lambda": 0.7},
                    "compose": {
                        "budget_tokens": 1500,
                        "snippet_chars": 420,
                        "tokenizer": "bert-base-multilingual-cased",
                    },
                    "guards": {
                        "min_support_docs": 2,
                        "min_support_score": 0.15,
                        "min_top1_score": 0.25,
                        "refuse_message": "Je ne peux pas répondre de façon fiable : support insuffisant. Peux-tu préciser la question ou partager une source ?",
                    },
                }

            # Import du pipeline uniquement ici, et seulement si on va l’utiliser
            try:
                from AGI_Evolutive.retrieval.rag5.pipeline import RAGPipeline

                self.rag = RAGPipeline(self.rag_cfg)
                # Seed initial depuis la mémoire (non bloquant)
                try:
                    for m in self.memory.get_recent_memories(n=2000):
                        txt = m.get("text")
                        if txt:
                            self.rag.add_document(
                                m.get("id", f"mem#{int(m.get('ts', 0))}"),
                                txt,
                                meta={"ts": m.get("ts"), "source_trust": 0.6},
                            )
                except Exception:
                    pass
            except Exception as e:
                # RAG restera désactivé si dépendances absentes
                self.rag = None
                if hasattr(self, "logger"):
                    self.logger.warning(
                        "RAG désactivé (dépendances manquantes ou erreur d'init): %s", e
                    )
        except Exception:
            # Ne jamais empêcher l'architecture de booter
            self.rag = None

        self.telemetry.log("init", "core", {"stage": "perception"})
        self.perception = PerceptionSystem(self, self.memory)

        self.telemetry.log("init", "core", {"stage": "reasoning"})
        self.reasoning = ReasoningSystem(self, self.memory, self.perception)

        self.telemetry.log("init", "core", {"stage": "goals"})
        self.goals = GoalSystem(self, self.memory, self.reasoning)

        self.telemetry.log("init", "core", {"stage": "metacognition"})
        self.metacognition = MetacognitiveSystem(self, self.memory, self.reasoning)

        self.telemetry.log("init", "core", {"stage": "emotions"})
        self.emotions = EmotionalSystem(self, self.memory, self.metacognition)

        self.telemetry.log("init", "core", {"stage": "learning"})
        self.learning = ExperientialLearning(self)

        self.telemetry.log("init", "core", {"stage": "creativity"})
        self.creativity = CreativitySystem(
            self, self.memory, self.reasoning, self.emotions, self.metacognition
        )

        self.telemetry.log("init", "core", {"stage": "world_model"})
        self.world_model = PhysicsEngine(self, self.memory)

        self.telemetry.log("init", "core", {"stage": "language"})
        self.language = SemanticUnderstanding(self, self.memory, intent_model=self.intent_model)

        self.concept_extractor = ConceptExtractor(self.memory)
        self.episodic_linker = EpisodicLinker(self.memory)

        self.action_interface = ActionInterface()
        self.perception_interface = PerceptionInterface()

        # Advanced subsystems
        self.style_profiler = StyleProfiler(persist_path="data/style_profiles.json")
        self.ontology = Ontology()
        self.beliefs = BeliefGraph(ontology=self.ontology)
        self.entity_linker = EntityLinker(self.ontology, self.beliefs)
        self.beliefs.set_entity_linker(self.entity_linker)
        self.scm = SCMStore(self.beliefs, self.ontology)
        self.simulator = CounterfactualSimulator(self.scm)
        self.planner = HTNPlanner(self.beliefs, self.ontology)
        self.user_model = UserModel()
        try:
            persona_tone = (self.user_model.describe().get("persona", {}) or {}).get("tone")
            if persona_tone:
                self.style_policy.update_persona_tone(persona_tone)
        except Exception:
            pass
        self.calibration = CalibrationMeter()
        self.calibration_abduction = CalibrationMeter(path="data/calibration_abduction.jsonl")
        self.calibration_concepts = CalibrationMeter(path="data/calibration_concepts.jsonl")
        self.calibration_causal = CalibrationMeter(path="data/calibration_causal.jsonl")
        self.calibration_plan = CalibrationMeter(path="data/calibration_plan.jsonl")
        self.novelty_detector = NoveltyDetector()
        self.abduction = AbductiveReasoner(self.beliefs, self.user_model)
        self.abduction.qengine = QuestionEngine(self.beliefs, self.user_model)
        self.reward_engine = RewardEngine(
            architecture=self,
            memory=self.memory,
            emotions=self.emotions,
            goals=self.goals,
            metacognition=self.metacognition,
            persist_dir="data",
        )

        self.autonomy = AutonomyManager(
            architecture=self,
            goal_system=self.goals,
            metacognition=self.metacognition,
            memory=self.memory,
            perception=self.perception,
            language=self.language,
        )

        def _apply_overrides(arch: "CognitiveArchitecture", ov: Dict[str, Any]) -> None:
            if not ov:
                return
            style_policy = getattr(arch, "style_policy", None)
            if style_policy and hasattr(style_policy, "params"):
                if "style.hedging" in ov:
                    style_policy.params["hedging"] = max(0.0, min(1.0, float(ov["style.hedging"])))

            if "learning.self_assess.threshold" in ov and hasattr(arch, "learning"):
                try:
                    threshold = float(ov["learning.self_assess.threshold"])
                    setattr(arch.learning, "self_assess_threshold", threshold)
                except Exception:
                    pass

            abduction = getattr(arch, "abduction", None)
            if not abduction:
                return
            if "abduction.tie_gap" in ov:
                setattr(abduction, "tie_gap", float(ov["abduction.tie_gap"]))
            if "abduction.weights.prior" in ov:
                setattr(abduction, "w_prior", float(ov["abduction.weights.prior"]))
            if "abduction.weights.boost" in ov:
                setattr(abduction, "w_boost", float(ov["abduction.weights.boost"]))
            if "abduction.weights.match" in ov:
                setattr(abduction, "w_match", float(ov["abduction.weights.match"]))

        def _arch_factory(overrides: Dict[str, Any]) -> "CognitiveArchitecture":
            fresh: "CognitiveArchitecture"
            try:
                fresh = self.__class__(boot_minimal=True)
            except Exception:
                fresh = self
            try:
                _apply_overrides(fresh, overrides or {})
            except Exception:
                pass
            return fresh

        self._arch_factory = _arch_factory
        self.promotions: Optional[PromotionManager]
        if not boot_minimal:
            self.self_improver = SelfImprover(
                arch_factory=self._arch_factory,
                memory=self.memory,
                question_manager=getattr(self, "question_manager", None),
                apply_overrides=lambda overrides: _apply_overrides(self, overrides),
            )
            self.promotions = self.self_improver.prom
            self.code_evolver: Optional[CodeEvolver] = getattr(self.self_improver, "code_evolver", None)
        else:
            self.self_improver = None
            self.promotions = None
            self.code_evolver = None

        # Persistence layer shared with Autopilot/logger
        existing_persistence = getattr(self, "persistence", None)
        if isinstance(existing_persistence, PersistenceManager):
            self.persistence = existing_persistence
        else:
            self.persistence = PersistenceManager(self)
        self.logger.persistence = self.persistence

        # Job manager for background/offloaded actions
        try:
            data_dir = cfg().get("DATA_DIR", "data")
        except Exception:
            data_dir = "data"
        self.jobs = JobManager(self, data_dir=data_dir)

        # Bind helper components
        self._bind_interfaces()
        self._bind_extractors()

        # Long-term modules
        self.evolution = EvolutionManager(data_dir="data")
        self.evolution.bind(
            architecture=self,
            memory=self.memory,
            metacog=self.metacognition,
            goals=self.goals,
            learning=self.learning,
            emotions=self.emotions,
            language=self.language,
        )

        # Scheduler runs background maintenance work (daemon thread)
        self.scheduler = None
        if not boot_minimal:
            self.scheduler = Scheduler(self, data_dir="data")
            self.scheduler.start()

        self.telemetry.log("ready", "core", {"status": "initialized"})
        self._cycle_counter = 0
        self._decay_period = 8

    # ------------------------------------------------------------------
    # Helpers
    def _bind_interfaces(self) -> None:
        policy = getattr(self, "policy", None)
        self.action_interface.bind(
            arch=self,
            goals=self.goals,
            policy=policy,
            memory=self.memory,
            metacog=self.metacognition,
            emotions=self.emotions,
            language=self.language,
            jobs=self.jobs,
        )
        self.perception_interface.bind(
            arch=self,
            memory=self.memory,
            metacog=self.metacognition,
            emotions=self.emotions,
            language=self.language,
        )

    def _bind_extractors(self) -> None:
        self.concept_extractor.bind(
            memory=self.memory,
            emotions=self.emotions,
            metacog=self.metacognition,
            language=self.language,
        )
        self.episodic_linker.bind(
            memory=self.memory,
            language=self.language,
            metacog=self.metacognition,
            emotions=self.emotions,
        )

    def _language_state_snapshot(self) -> Dict[str, Any]:
        language = getattr(self, "language", None)
        dialogue = None
        if language is not None:
            dialogue = getattr(language, "state", None)
            dialogue = getattr(language, "dialogue_state", dialogue)
        return {
            "beliefs": getattr(self, "beliefs", None),
            "self_model": getattr(self, "self_model", None),
            "dialogue": dialogue,
            "world": getattr(self, "world_model", None),
            "memory": getattr(self, "memory", None),
        }

    def _predicate_registry_for_state(self, state: Dict[str, Any]) -> Dict[str, Callable[..., bool]]:
        policy = getattr(self, "policy", None)
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

    def _resolve_nlg_hint_applier(self) -> Callable[[str, str], str]:
        renderer = getattr(self, "renderer", None)
        if renderer is not None and hasattr(renderer, "apply_action_hint"):
            fn = getattr(renderer, "apply_action_hint")
            if callable(fn):
                return lambda text, hint: fn(text, hint)

        try:
            from AGI_Evolutive.language.renderer import _apply_action_hint

            return lambda text, hint: _apply_action_hint(text, hint)
        except Exception:
            return lambda text, hint: text

    def reindex_rag_from_memory(self, limit: int = 10000) -> int:
        """(Ré)indexe des souvenirs texte récents dans le RAG. Retourne le nb de docs ajoutés."""
        if not getattr(self, "rag", None):
            return 0
        n = 0
        try:
            for m in self.memory.get_recent_memories(n=limit):
                txt = m.get("text")
                if txt:
                    self.rag.add_document(
                        m.get("id", f"mem#{int(m.get('ts', 0))}"),
                        txt,
                        meta={"ts": m.get("ts"), "source_trust": 0.6},
                    )
                    n += 1
        except Exception:
            pass
        return n

    # ------------------------------------------------------------------
    # Status & reporting
    def _present_subsystems(self) -> Dict[str, bool]:
        names = [
            "memory",
            "perception",
            "reasoning",
            "goals",
            "metacognition",
            "emotions",
            "learning",
            "creativity",
            "world_model",
            "language",
        ]
        return {name: getattr(self, name, None) is not None for name in names}

    def get_cognitive_status(self) -> Dict[str, Any]:
        wm_load = 0.0
        try:
            wm = getattr(self.memory, "working_memory", None)
            if wm and hasattr(wm, "__len__"):
                wm_load = min(len(wm) / 10.0, 1.0)
        except Exception:
            wm_load = 0.0

        return {
            "uptime_s": int(time.time() - self.start_time),
            "global_activation": float(self.global_activation),
            "working_memory_load": float(wm_load),
            "subsystems": self._present_subsystems(),
            "style_policy": self.style_policy.as_dict(),
            "goal_focus": self.goal_dag.choose_next_goal(),
        }

    def diagnostic_snapshot(self, tail: int = 30) -> Dict[str, Any]:
        return {
            "status": self.get_cognitive_status(),
            "tail": self.telemetry.tail(tail),
        }

    def summarize_beliefs(self, timeframe: str = "daily") -> Dict[str, Any]:
        beliefs = getattr(self, "beliefs", None)
        if not beliefs:
            return {}
        try:
            summary = beliefs.latest_summary()
            if timeframe:
                return summary.get(timeframe, {})
            return summary
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Cycle
    def cycle(
        self,
        user_msg: Optional[str] = None,
        inbox_docs=None,
        user_id: str = "default",
    ) -> str:
        self.last_user_id = user_id or "default"

        if not user_msg:
            self._tick_background_systems()
            return self.last_output_text

        self._cycle_counter += 1
        if self._cycle_counter % self._decay_period == 0:
            self._apply_belief_decay()

        try:
            if self.self_improver and self.self_improver.try_promote_from_reply(user_msg):
                return "✅ Challenger promu. Les nouveaux paramètres sont actifs."
        except Exception:
            pass

        if isinstance(user_msg, str):
            normalized = user_msg.strip().lower()
            if normalized in {"améliore-toi", "self-improve", "optimize"}:
                cid = None
                try:
                    if self.self_improver:
                        cid = self.self_improver.run_cycle(n_candidates=4)
                except Exception:
                    cid = None
                if cid:
                    return (
                        "J’ai un challenger candidat ({cid}). Je te demande validation avant promotion.".format(
                            cid=cid
                        )
                    )
                return "Aucun challenger n’a surclassé le champion sur les métriques définies."

        trimmed = (user_msg or "").strip()
        trimmed_lower = trimmed.lower()
        explicit_mode_cmd = trimmed_lower.startswith("/mode")
        detected_mode = None
        try:
            detected_mode = self.style_policy.detect_mode_command(trimmed_lower)
        except Exception:
            detected_mode = None
        if detected_mode:
            phrase_command = trimmed_lower in {
                f"mode {detected_mode}",
                f"mode {detected_mode}.",
                f"mode {detected_mode}!",
                f"mode {detected_mode}?",
            }
            if phrase_command:
                explicit_mode_cmd = True
            self.style_policy.set_mode(detected_mode, persona_tone=self.style_policy.persona_tone)
            if explicit_mode_cmd and len(trimmed.split()) <= 2:
                ack = f"✅ Mode de communication réglé sur « {detected_mode} »."
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="mode_switch",
                            content=detected_mode,
                            metadata={"source": "user"},
                        )
                except Exception:
                    pass
                self.last_output_text = ack
                return ack

        try:
            persona_tone = (self.user_model.describe().get("persona", {}) or {}).get("tone")
            if persona_tone:
                self.style_policy.update_persona_tone(persona_tone)
        except Exception:
            pass

        try:
            self.intent_model.observe_user_message(user_msg)
        except Exception:
            pass

        self._maybe_update_calibration(user_msg)

        self.telemetry.log("input", "language", {"text": user_msg})

        try:
            self.autonomy.notify_user_activity()
        except Exception:
            pass

        try:
            self.style_profiler.observe(self.last_user_id, user_msg)
        except Exception:
            pass

        try:
            context = {
                "last_assistant_output": self.last_output_text,
                "active_goal_id": getattr(self.goals, "active_goal_id", None),
            }
            self.reward_engine.ingest_user_message(
                self.last_user_id, user_msg, context=context, channel="chat"
            )
        except Exception:
            pass

        def _looks_like_abduction(s: str) -> bool:
            s = (s or "").lower()
            return any(
                k in s
                for k in ["devine", "pourquoi", "hypothèse", "à ton avis", "raison", "résous l'énigme"]
            )

        abduction_result: Optional[Dict[str, Any]] = None
        if _looks_like_abduction(user_msg):
            abduction_result = self._handle_abduction_request(user_msg)
        else:
            if self._looks_like_causal(trimmed_lower):
                causal_reply = self._handle_causal(user_msg)
                if causal_reply:
                    self.last_output_text = causal_reply
                    return causal_reply
            if self._looks_like_plan(trimmed_lower):
                plan_reply = self._handle_plan(user_msg)
                if plan_reply:
                    self.last_output_text = plan_reply
                    return plan_reply

        surface = user_msg
        hints = {}
        try:
            hints = self.style_policy.adapt_from_instruction(user_msg)
        except Exception:
            hints = {}
        try:
            parsed = self.language.parse_utterance(
                user_msg,
                context={
                    "style_hints": hints,
                    "style_mode": self.style_policy.current_mode,
                    "user_intents": self.intent_model.as_constraints(),
                },
            )
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            pass

        novelty_score = 0.0
        novelty_flag = False
        try:
            novelty_score, novelty_flag = self.novelty_detector.assess(surface, update=True)
            if novelty_flag and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="novel_case",
                    content=surface[:160],
                    metadata={"score": float(novelty_score), "user_id": self.last_user_id},
                )
            try:
                self.logger.write(
                    "novelty.assessment",
                    score=float(novelty_score),
                    flagged=bool(novelty_flag),
                    text=surface[:120],
                )
            except Exception:
                pass
        except Exception:
            novelty_score = 0.0
            novelty_flag = False

        self._emit_structured_memories(surface)

        # Capture d'une confirmation utilisateur pour valider un apprentissage récent
        try:
            low = (user_msg or "").lower()
            if any(k in low for k in ["oui", "c'est correct", "exact"]) and hasattr(self, "memory"):
                recents = self.memory.get_recent_memories(50)
                concept = None
                for item in reversed(recents):
                    if item.get("kind") == "validation_request":
                        concept = (item.get("metadata") or {}).get("concept")
                        if concept:
                            break
                if concept:
                    self._record_skill(concept)
            else:
                refusal_markers = (
                    "pas correct",
                    "incorrect",
                    "c'est faux",
                    "ce n'est pas correct",
                    "je ne suis pas d'accord",
                    "pas vrai",
                    "mauvais",
                )
                refuse_short = low.strip() in {"non", "nope", "pas vraiment", "absolument pas"}
                refusal_hit = refuse_short or any(m in low for m in refusal_markers)
                if refusal_hit and hasattr(self, "memory"):
                    recents = self.memory.get_recent_memories(50)
                    concept = None
                    last_request = None
                    for item in reversed(recents):
                        if item.get("kind") == "validation_request":
                            last_request = item
                            concept = (item.get("metadata") or {}).get("concept")
                            if concept:
                                break
                    if concept:
                        try:
                            self.memory.add_memory(
                                {
                                    "kind": "validation_request:refusé",
                                    "content": f"Validation refusée pour {concept}",
                                    "metadata": {
                                        "concept": concept,
                                        "source": "user_feedback",
                                        "raw": user_msg[:160],
                                        "request_id": (last_request or {}).get("id"),
                                    },
                                    "ts": time.time(),
                                }
                            )
                        except Exception:
                            pass
                        try:
                            if not getattr(self, "concept_recognizer", None):
                                from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer

                                self.concept_recognizer = ConceptRecognizer(self)
                            evidence_payload = {}
                            mem = getattr(self, "memory", None)
                            if mem and hasattr(mem, "find_recent"):
                                ev = mem.find_recent(
                                    kind="concept_candidate",
                                    since_sec=3600 * 24,
                                    where={"label": concept},
                                ) or {}
                                if isinstance(ev, dict):
                                    evidence_payload = ev.get("evidence", {}) or {}
                                elif isinstance(ev, list) and ev:
                                    first = ev[0]
                                    if isinstance(first, dict):
                                        evidence_payload = first.get("evidence", {}) or {}
                            if getattr(self, "concept_recognizer", None):
                                self.concept_recognizer.learn_from_rejection(
                                    kind="concept",
                                    label=concept,
                                    evidence=evidence_payload,
                                    penalty=0.6,
                                )
                        except Exception:
                            pass
        except Exception:
            pass

        reason_out: Dict[str, Any] = {}
        if abduction_result:
            reason_out = dict(abduction_result.get("reason_out") or {})
        else:
            try:
                reasoning_context = {
                    "inbox_docs": inbox_docs,
                    "user_intents": self.intent_model.as_constraints(),
                    "style_mode": self.style_policy.current_mode,
                }
                reason_out = self.reasoning.reason_about(surface, context=reasoning_context)
            except Exception as exc:
                self.logger.write("reasoning.error", error=str(exc), user_msg=surface)
                reason_out = {
                    "summary": "Raisonnement basique uniquement (fallback).",
                    "chosen_hypothesis": "clarifier intention + proposer 1 test",
                    "tests": ["proposer 2 options et valider"],
                    "final_confidence": 0.5,
                    "appris": ["garder une trace même en cas d'erreur"],
                    "prochain_test": "valider l'option la plus utile",
                }

        ask_prompts: List[str] = []
        abstain = False
        calibration_domain = None
        raw_confidence = float(reason_out.get("final_confidence", 0.5))
        adjusted_confidence = max(0.0, min(1.0, raw_confidence))

        if not abduction_result:
            calibration_domain = "planning" if reason_out.get("tests") or reason_out.get("prochain_test") else "decision"
            if novelty_flag:
                adjusted_confidence *= max(0.4, 1.0 - 0.35 * novelty_score)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            reason_out["final_confidence"] = adjusted_confidence
            try:
                abstain = self.calibration.should_abstain(calibration_domain, adjusted_confidence)
            except Exception:
                abstain = False
            if novelty_flag and adjusted_confidence < 0.75:
                abstain = True
            event_id = None
            try:
                meta = {
                    "domain": calibration_domain,
                    "novelty": float(novelty_score),
                    "abstain": bool(abstain),
                    "raw_confidence": float(raw_confidence),
                }
                event_id = self.calibration.log_prediction(
                    domain=calibration_domain,
                    p=adjusted_confidence,
                    meta=meta,
                )
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="calibration_observation",
                        content=f"{calibration_domain or 'decision'}_pred",
                        metadata={
                            "event_id": event_id,
                            "domain": calibration_domain,
                            "p": float(adjusted_confidence),
                        },
                    )
            except Exception:
                event_id = None
            if abstain:
                reason_out.setdefault("appris", []).append(
                    "Appliquer un refus calibré quand la confiance est insuffisante."
                )
                ask_prompts.append(
                    "Confiance trop faible → peux-tu préciser ton objectif ou les contraintes clés ?"
                )
                reason_out["summary"] = (
                    reason_out.get("summary", "")
                    + " | ⚠️ abstention calibrée (demander précisions)."
                )
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="abstain",
                            content=surface[:160],
                            metadata={"domain": calibration_domain, "confidence": adjusted_confidence},
                        )
                except Exception:
                    pass
        try:
            if not abduction_result and hasattr(self, "question_manager") and self.question_manager:
                severity = max(0.0, 1.0 - adjusted_confidence)
                if severity > 0.45:
                    explicit_q = ask_prompts[0] if ask_prompts else None
                    self.question_manager.record_information_need(
                        "goal_focus",
                        severity,
                        metadata={"source": "confidence", "user_msg": surface[:160]},
                        explicit_question=explicit_q,
                    )
        except Exception:
            pass
        if novelty_flag:
            if abduction_result:
                adjusted_confidence = max(
                    0.0,
                    min(1.0, adjusted_confidence * max(0.4, 1.0 - 0.35 * novelty_score)),
                )
                reason_out["final_confidence"] = adjusted_confidence
            reason_out.setdefault("appris", []).append(
                "Détecter les cas atypiques et demander un éclairage supplémentaire."
            )
            if "Cas inhabituel détecté → partage un exemple ou le contexte exact." not in ask_prompts:
                ask_prompts.append(
                    "Cas inhabituel détecté → partage un exemple ou le contexte exact."
                )
            try:
                if hasattr(self, "question_manager") and self.question_manager:
                    self.question_manager.record_information_need(
                        "evidence",
                        max(0.4, min(1.0, float(novelty_score) or 0.5)),
                        metadata={"source": "novelty", "user_msg": surface[:160]},
                    )
            except Exception:
                pass

        apprentissages = [
            "associer récompense sociale ↔ style",
            "tenir un journal d'épisodes de raisonnement",
        ] + list(reason_out.get("appris", []))

        next_test = reason_out.get("prochain_test") or "-"
        if abstain:
            next_test = "clarifier avec toi les contraintes et objectifs avant d'avancer"

        base_besoins = ["confirmer si tu veux patch immédiat ou plan en étapes"]
        besoins: List[str] = []
        for item in base_besoins + ask_prompts:
            if not item:
                continue
            if item not in besoins:
                besoins.append(item)

        contract = ensure_contract(
            {
                "hypothese_choisie": reason_out.get("chosen_hypothesis", "clarifier intention"),
                "incertitude": float(max(0.0, min(1.0, 1.0 - adjusted_confidence))),
                "prochain_test": next_test,
                "appris": apprentissages,
                "besoin": besoins,
            }
        )

        base_text = self._generate_base_text(surface, reason_out)
        nlg_state = self._language_state_snapshot()
        predicate_registry = self._predicate_registry_for_state(nlg_state)
        nlg_context = NLGContext(base_text, self._resolve_nlg_hint_applier())
        try:
            apply_mai_bids_to_nlg(nlg_context, nlg_state, predicate_registry)
        except Exception:
            pass
        applied_hints = nlg_context.applied_hints()
        if applied_hints:
            reason_out["applied_hints"] = applied_hints
        response = format_agent_reply(nlg_context.text, **contract)

        if abduction_result:
            response = abduction_result.get("response", response)

        try:
            response = self.style_profiler.rewrite_to_match(response, self.last_user_id)
        except Exception:
            pass

        self.last_output_text = response

        try:
            if hasattr(self.memory, "store_interaction"):
                self.memory.store_interaction(
                    {
                        "ts": time.time(),
                        "user": user_msg,
                        "agent": response,
                        "lang_state": getattr(
                            getattr(self.language, "state", None), "to_dict", lambda: {}
                        )(),
                    }
                )
        except Exception:
            pass

        self.logger.write(
            "dialogue.turn",
            user_msg=user_msg,
            surface=surface,
            hypothesis=contract["hypothese_choisie"],
            incertitude=contract["incertitude"],
            test=contract["prochain_test"],
            reward=extract_social_reward(user_msg).get("reward", 0.0),
            style=self.style_policy.as_dict(),
            reason_summary=reason_out.get("summary", ""),
        )

        try:
            mem = self.memory.get_recent_memories(60)
            self.user_model.ingest_memories(mem)
        except Exception:
            pass

        self._tick_background_systems()
        return response

    def _handle_abduction_request(self, user_msg: str) -> Dict[str, Any]:
        reason_out: Dict[str, Any] = {
            "summary": "Aucune hypothèse abductive trouvée.",
            "chosen_hypothesis": "aucune",
            "final_confidence": 0.0,
            "prochain_test": None,
            "appris": ["reconnaître une demande abductive"],
        }
        try:
            hyps = self.abduction.generate(user_msg)
        except Exception:
            hyps = []
        if not hyps:
            return {
                "response": "Je manque d'indices pour formuler une hypothèse utile.",
                "reason_out": reason_out,
            }

        top = hyps[0]
        score = float(getattr(top, "score", 0.0))
        reason_out.update(
            {
                "summary": top.explanation or "Hypothèse générée via abduction.",
                "chosen_hypothesis": top.label,
                "final_confidence": score,
                "prochain_test": getattr(top, "ask_next", None),
            }
        )
        appris = list(reason_out.get("appris") or [])
        if getattr(top, "priors", None):
            appris.append("mobiliser les priors abductifs")
        reason_out["appris"] = appris
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="hypothesis",
                    content=top.label,
                    metadata={
                        "score": top.score,
                        "explanation": top.explanation,
                        "priors": top.priors,
                    },
                )
        except Exception:
            pass

        ev_id = None
        try:
            ev_id = self.calibration_abduction.log_prediction(
                domain="abduction", p=float(top.score), meta={"label": top.label}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="abduction_pred",
                    metadata={
                        "event_id": ev_id,
                        "label": top.label,
                        "p": float(top.score),
                        "domain": "abduction",
                    },
                )
        except Exception:
            ev_id = None

        if top.ask_next:
            handled = False
            try:
                qm = getattr(self, "question_manager", None)
                if qm and hasattr(qm, "add_question"):
                    qm.add_question(top.ask_next)
                    handled = True
            except Exception:
                handled = False
            if not handled:
                try:
                    if hasattr(self.memory, "add_memory"):
                        self.memory.add_memory(
                            kind="question_active",
                            content=top.ask_next,
                            metadata={"source": "abduction"},
                        )
                except Exception:
                    pass
            reason_out["summary"] = top.explanation or "Question de clarification abductive."
            return {"response": f"{top.ask_next}", "reason_out": reason_out}

        conf = int(round(score * 100))
        response = (
            f"Mon hypothèse la plus probable : **{top.label}** ({conf}% confiance). "
            "Je peux réviser si tu me donnes un indice contraire."
        )
        return {"response": response, "reason_out": reason_out}

    def _maybe_update_calibration(self, user_msg: Optional[str]) -> None:
        t = (user_msg or "").strip().lower()
        if t not in {"oui", "non", "exact", "c'est correct"}:
            return
        try:
            recents = []
            if hasattr(self.memory, "get_recent_memories"):
                recents = self.memory.get_recent_memories(50)
            ev_id = None
            event_meta: Dict[str, Any] = {}
            for item in reversed(recents):
                if item.get("kind") == "calibration_observation":
                    event_meta = item.get("metadata") or {}
                    ev_id = event_meta.get("event_id")
                    if ev_id:
                        break
            if ev_id:
                success = t in {"oui", "exact", "c'est correct"}
                domain = event_meta.get("domain", "abduction")
                self.calibration.log_outcome(ev_id, success=success)
                delta = self.calibration.suggested_hedging_delta(domain=domain)
                if hasattr(self.style_policy, "params"):
                    hedging = self.style_policy.params.get("hedging", 0.3)
                    self.style_policy.params["hedging"] = max(0.0, min(1.0, hedging + delta))
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="calibration_feedback",
                        content=t,
                        metadata={"event_id": ev_id, "delta_hedging": delta, "domain": domain},
                    )
        except Exception:
            pass

    def _emit_structured_memories(self, text: str) -> None:
        if not text or not hasattr(self.memory, "add_memory"):
            return
        try:
            entities = re.findall(r"\b[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][\w'’\-]{2,}\b", text)
            seen = set()
            for ent in entities[:6]:
                key = ent.lower()
                if key in seen:
                    continue
                seen.add(key)
                linked = self.entity_linker.link(ent)
                self.memory.add_memory(
                    kind="entity_detected",
                    content=ent,
                    metadata={"type": linked["type"], "canonical": linked["canonical"]},
                )
        except Exception:
            pass

        try:
            for match in re.finditer(
                r"([A-Za-zÀ-ÖØ-öø-ÿ'’\-]{2,})\s+(est|sont|sera|éta(?:is|it|ient))\s+([^\.;]+)",
                text,
            ):
                subject = match.group(1).strip()
                relation = match.group(2).strip().lower()
                value = match.group(3).strip()
                if not subject or not value:
                    continue
                subj_link = self.entity_linker.link(subject)
                val_link = self.entity_linker.link(value)
                self.memory.add_memory(
                    kind="fact_extracted",
                    content=f"{subject} {relation} {value}",
                    metadata={
                        "subject": subj_link["canonical"],
                        "relation": relation,
                        "value": val_link["canonical"],
                        "polarity": +1,
                    },
                )
        except Exception:
            pass

    def _looks_like_causal(self, text: str) -> bool:
        if not text:
            return False
        cues = [
            "pourquoi",
            "cause",
            "causal",
            "que se passerait-il",
            "que se passerait il",
        ]
        if any(cue in text for cue in cues):
            return True
        return " si " in text and "alors" in text

    def _looks_like_plan(self, text: str) -> bool:
        if not text:
            return False
        cues = ["planifie", "planifier", "plan", "comment atteindre", "objectif"]
        return any(cue in text for cue in cues)

    def _parse_cause_effect(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"si\s+(.+?)\s+alors\s+(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    def _handle_causal(self, user_msg: str) -> Optional[str]:
        if not getattr(self, "simulator", None):
            return None
        cause, effect = self._parse_cause_effect(user_msg)
        query = {
            "cause": cause,
            "effect": effect,
            "scenario": {"utterance": user_msg},
        }
        try:
            report = self.simulator.run(query)
        except Exception as exc:
            return f"Je n'ai pas pu exécuter la simulation causale ({exc})."

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="causal_query",
                    content=user_msg[:160],
                    metadata={"cause": cause, "effect": effect},
                )
        except Exception:
            pass

        probability = 0.7 if report.supported else 0.35
        event_id = None
        try:
            event_id = self.calibration_causal.log_prediction(
                "causal", probability, meta={"cause": cause, "effect": effect}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="causal_pred",
                    metadata={"event_id": event_id, "p": probability, "domain": "causal"},
                )
        except Exception:
            event_id = None

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="counterfactual_result",
                    content=f"{cause or 'cause?'} → {effect or 'effet?'}",
                    metadata={
                        "supported": report.supported,
                        "evidence": report.evidence,
                        "intervention": report.intervention,
                    },
                )
        except Exception:
            pass

        evidence_text = (
            "J'observe un lien causal existant dans ma base." if report.supported else "Je ne possède pas de lien causal établi pour cette relation."
        )
        sim_texts = [sim.get("outcome", "") for sim in report.simulations if sim]
        if sim_texts:
            evidence_text += " " + sim_texts[0][:160]
        return evidence_text

    def _handle_plan(self, user_msg: str) -> Optional[str]:
        if not getattr(self, "planner", None):
            return None
        goal = user_msg
        match = re.search(r"(?:planifie|planifier|plan pour|comment atteindre)\s+(.+)", user_msg, re.IGNORECASE)
        if match:
            goal = match.group(1).strip()
        steps = self.planner.plan("diagnostic_general", context={"goal": goal}) or [
            f"Clarifier le résultat attendu pour « {goal} ».",
            "Identifier ressources et contraintes majeures.",
            "Découper en trois actions concrètes et dater la première.",
        ]

        probability = min(0.95, 0.55 + 0.05 * len(steps))
        event_id = None
        try:
            event_id = self.calibration_plan.log_prediction(
                "plan", probability, meta={"goal": goal}
            )
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="calibration_observation",
                    content="plan_pred",
                    metadata={"event_id": event_id, "p": probability, "domain": "plan"},
                )
        except Exception:
            event_id = None

        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="plan_created",
                    content=goal[:160],
                    metadata={"steps": steps},
                )
        except Exception:
            pass

        numbered = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps))
        return f"Plan proposé pour « {goal} » :\n{numbered}"

    def _apply_belief_decay(self) -> None:
        try:
            now = time.time()
            decayed = 0
            for belief in self.beliefs.iter_beliefs():
                age = now - belief.updated_at
                if age < 180 or belief.confidence <= 0.2:
                    continue
                belief.confidence = max(0.0, belief.confidence - 0.02)
                belief.updated_at = now
                decayed += 1
            if decayed:
                self.beliefs.flush()
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="belief_decay",
                        content=f"{decayed} croyances ajustées",
                        metadata={"timestamp": now},
                    )
        except Exception:
            pass

    def _tick_background_systems(self) -> None:
        try:
            if getattr(self, "jobs", None):
                self.jobs.drain_to_memory(self.memory)
        except Exception:
            pass

        try:
            if self.perception_interface:
                self.perception_interface.step()
        except Exception:
            pass

        try:
            if self.emotions and hasattr(self.emotions, "step"):
                self.emotions.step()
        except Exception:
            pass

        try:
            if self.action_interface:
                self.action_interface.step()
        except Exception:
            pass

        try:
            self.concept_extractor.step(self.memory)
        except Exception:
            pass

        try:
            self.episodic_linker.step(self.memory)
        except Exception:
            pass

        try:
            self.autonomy.tick()
        except Exception:
            pass

        try:
            if time.time() - self._last_intent_decay > 600:
                self.intent_model.decay()
                self._last_intent_decay = time.time()
        except Exception:
            pass

        try:
            if self.memory and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    "dialog_turn",
                    {
                        "t": time.time(),
                        "user_id": self.last_user_id,
                        "assistant_msg": self.last_output_text,
                    },
                )
        except Exception:
            pass

    def _record_skill(
        self,
        concept: str,
        *,
        source: str = "learn_concept",
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Enregistre un 'skill/concept' appris et consolide la connaissance :
          - Persistance JSON (data/skills.json)
          - Mémoire milestone (tags: lesson, concept:<name>)
          - Ontology/Beliefs: crée le nœud concept:*, evidence 'defined_by'
          - Consolidator: ajoute une 'lesson' (résumé court)
          - Émet un événement 'virtue_learned' (si concept fait sens pour la persona)
        Robuste: aucun crash si un composant (ontology/beliefs/consolidator/voice) est absent.
        """
        import os, json, time, traceback

        t0 = time.time()
        out = {"ok": True, "concept": concept, "path": None, "errors": []}
        concept_norm = (concept or "").strip()
        if not concept_norm:
            return {"ok": False, "concept": concept, "errors": ["empty_concept"]}

        # ---------------------------------------
        # 1) Persistance JSON (skills.json)
        # ---------------------------------------
        skills_path = getattr(self, "skills_path", os.path.join("data", "skills.json"))
        os.makedirs(os.path.dirname(skills_path) or ".", exist_ok=True)

        try:
            skills = {}
            if os.path.exists(skills_path):
                with open(skills_path, "r", encoding="utf-8") as f:
                    try:
                        skills = json.load(f) or {}
                    except Exception:
                        skills = {}

            entry = skills.get(concept_norm) or {}
            entry.update({
                "name": concept_norm,
                "acquired": True,
                "last_update": t0,
                "source": source,
                "confidence": float(confidence) if confidence is not None else entry.get("confidence", 1.0),
                "meta": {**(entry.get("meta", {}) or {}), **(metadata or {})},
            })
            skills[concept_norm] = entry

            with open(skills_path, "w", encoding="utf-8") as f:
                json.dump(skills, f, ensure_ascii=False, indent=2)
            out["path"] = skills_path
        except Exception as e:
            out["ok"] = False
            out["errors"].append(f"skills_persist:{e}")
            # on continue malgré tout

        # ---------------------------------------
        # 2) Mémoire milestone (trace datée)
        # ---------------------------------------
        try:
            if hasattr(self, "memory") and getattr(self, "memory"):
                self.memory.add_memory({
                    "kind": "milestone",
                    "text": f"Compréhension de {concept_norm} validée",
                    "ts": t0,
                    "tags": ["lesson", f"concept:{concept_norm}"],
                    "metadata": {
                        "concept": concept_norm,
                        "source": source,
                        "confidence": confidence,
                        **(metadata or {})
                    }
                })
        except Exception as e:
            out["errors"].append(f"milestone:{e}")

        # ---------------------------------------
        # 3) Ontology + Beliefs (evidence)
        # ---------------------------------------
        try:
            # Ontology
            if hasattr(self, "ontology") and getattr(self, "ontology"):
                # API tolérante: add_entity(id, attrs=...) ou add_entity(id, **attrs)
                try:
                    self.ontology.add_entity(f"concept:{concept_norm}", attrs={"kind": "concept", "label": concept_norm, "source": source})
                except TypeError:
                    self.ontology.add_entity(f"concept:{concept_norm}", kind="concept", label=concept_norm, source=source)
            # Beliefs
            if hasattr(self, "beliefs") and getattr(self, "beliefs"):
                ev = None
                try:
                    from AGI_Evolutive.beliefs.graph import Evidence
                    # Evidence.new(type, via, info, weight)
                    ev = Evidence.new("action", source, f"appris:{concept_norm}", weight=0.8)
                except Exception:
                    ev = None
                # tolérance d’API: add_fact(...) ou add_evidence(...)
                if hasattr(self.beliefs, "add_fact"):
                    self.beliefs.add_fact(subject=f"concept:{concept_norm}",
                                          predicate="defined_by",
                                          obj=source,
                                          evidence=ev)
                elif hasattr(self.beliefs, "add_evidence"):
                    self.beliefs.add_evidence(subject=f"concept:{concept_norm}",
                                              predicate="defined_by",
                                              obj=source,
                                              weight=0.8)
                if hasattr(self.beliefs, "flush"):
                    self.beliefs.flush()
        except Exception as e:
            out["errors"].append(f"ontology_beliefs:{e}")

        # ---------------------------------------
        # 4) Consolidator: lesson/synthèse
        # ---------------------------------------
        try:
            if hasattr(self, "consolidator") and getattr(self, "consolidator"):
                summary = f"Concept : {concept_norm} — défini et validé (source={source}). " \
                          f"Confiance={confidence if confidence is not None else 'n/a'}."
                st = getattr(self.consolidator, "state", None)
                if st is not None:
                    lessons = st.setdefault("lessons", [])
                    lessons.append({
                        "topic": concept_norm,
                        "summary": summary,
                        "sources": [f"goal:{source}"] if source else [],
                        "ts": t0,
                        "tags": ["lesson", f"concept:{concept_norm}"]
                    })
                    # _save() si dispo
                    if hasattr(self.consolidator, "_save"):
                        self.consolidator._save()
        except Exception as e:
            out["errors"].append(f"consolidator:{e}")

        # ---------------------------------------
        # 5) Événement pour la persona/voix (optionnel, non bloquant)
        #    -> permettra au Proposer de faire évoluer persona.values/tone
        # ---------------------------------------
        try:
            if hasattr(self, "memory") and getattr(self, "memory"):
                # Si c’est une “vertu”/valeur probable, on émet un hook
                virtues = {"empathy", "compassion", "kindness", "honesty", "precision"}
                if concept_norm.lower() in virtues:
                    self.memory.add_memory({
                        "kind": "virtue_learned",
                        "value": concept_norm.lower(),
                        "ts": t0,
                        "tags": ["persona_hook", f"concept:{concept_norm}"]
                    })
            # petit coup de pouce voix (si présent) – non bloquant
            if hasattr(self, "voice_profile") and getattr(self, "voice_profile"):
                try:
                    if concept_norm.lower() in {"empathy", "compassion", "kindness"}:
                        self.voice_profile.bump("warmth", +0.03)
                        self.voice_profile.bump("emoji", +0.02)
                    elif concept_norm.lower() in {"precision", "rigor", "rigueur"}:
                        self.voice_profile.bump("conciseness", +0.02)
                        self.voice_profile.bump("analytical", +0.02)
                except Exception:
                    pass
        except Exception as e:
            out["errors"].append(f"voice_event:{e}")

        try:
            if not getattr(self, "concept_recognizer", None):
                from AGI_Evolutive.knowledge.concept_recognizer import ConceptRecognizer
                self.concept_recognizer = ConceptRecognizer(self)
            mem = getattr(self, "memory", None)
            ev = {}
            if mem and hasattr(mem, "find_recent"):
                ev = mem.find_recent(kind="concept_candidate", since_sec=3600 * 24, where={"label": concept_norm}) or {}
            evidence_payload = {}
            if isinstance(ev, dict):
                evidence_payload = ev.get("evidence", {}) or {}
            elif isinstance(ev, list) and ev:
                first = ev[0]
                if isinstance(first, dict):
                    evidence_payload = first.get("evidence", {}) or {}
            if getattr(self, "concept_recognizer", None):
                self.concept_recognizer.learn_from_confirmation(
                    kind="concept",
                    label=concept_norm,
                    evidence=evidence_payload,
                    reward=0.85,
                )
        except Exception:
            pass

        out["duration_s"] = round(time.time() - t0, 3)
        # Si on a rencontré des erreurs non critiques, on reste ok=True mais on les remonte
        return out

    # ------------------------------------------------------------------
    def _generate_base_text(self, surface: str, reason_out: Dict[str, Any]) -> str:
        status = self.get_cognitive_status()
        status_line = (
            f"⏱️{status['uptime_s']}s | 🔋act={status['global_activation']:.2f} | "
            f"🧠wm={status['working_memory_load']:.2f}"
        )
        focus = status["goal_focus"]
        focus_line = (
            f"🎯focus:{focus['id']} (EVI={focus['evi']:.2f}, prog={focus['progress']:.2f})"
            if isinstance(focus, dict)
            else "🎯focus: n/a"
        )
        summary = reason_out.get("summary", "")
        return f"Reçu: {surface}\n{status_line}\n{focus_line}\n🧠 {summary}"
