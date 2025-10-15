import time
from typing import Any, Dict, Optional, List

from AGI_Evolutive.autonomy import AutonomyManager
from AGI_Evolutive.beliefs.graph import BeliefGraph, Evidence
from AGI_Evolutive.cognition.evolution_manager import EvolutionManager
from AGI_Evolutive.cognition.reward_engine import RewardEngine
from AGI_Evolutive.core.telemetry import Telemetry
from AGI_Evolutive.creativity import CreativitySystem
from AGI_Evolutive.emotions import EmotionalSystem
from AGI_Evolutive.goals import GoalSystem
from AGI_Evolutive.goals.dag_store import GoalDAG
from AGI_Evolutive.io.action_interface import ActionInterface
from AGI_Evolutive.io.perception_interface import PerceptionInterface
from AGI_Evolutive.language import SemanticUnderstanding
from AGI_Evolutive.language.policy import StylePolicy
from AGI_Evolutive.language.social_reward import extract_social_reward
from AGI_Evolutive.language.style_profiler import StyleProfiler
from AGI_Evolutive.learning import ExperientialLearning
from AGI_Evolutive.memory import MemorySystem
from AGI_Evolutive.memory.concept_extractor import ConceptExtractor
from AGI_Evolutive.memory.episodic_linker import EpisodicLinker
from AGI_Evolutive.metacog.calibration import CalibrationMeter, NoveltyDetector
from AGI_Evolutive.metacognition import MetacognitiveSystem
from AGI_Evolutive.models.user import UserModel
from AGI_Evolutive.perception import PerceptionSystem
from AGI_Evolutive.reasoning import ReasoningSystem
from AGI_Evolutive.reasoning.abduction import AbductiveReasoner, Hypothesis
from AGI_Evolutive.runtime.logger import JSONLLogger
from AGI_Evolutive.runtime.response import ensure_contract, format_agent_reply
from AGI_Evolutive.runtime.scheduler import Scheduler
from AGI_Evolutive.world_model import PhysicsEngine
from AGI_Evolutive.self_improver import SelfImprover
from AGI_Evolutive.self_improver.promote import PromotionManager


class CognitiveArchitecture:
    """Central coordinator for the agent's cognitive subsystems."""

    def __init__(self, boot_minimal: bool = False):
        self.boot_minimal = boot_minimal
        # Observability
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.telemetry = Telemetry()
        self.style_policy = StylePolicy()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Global state
        self.global_activation = 0.5
        self.start_time = time.time()
        self.reflective_mode = True
        self.last_output_text = "OK"
        self.last_user_id = "default"

        # Core subsystems
        self.telemetry.log("init", "core", {"stage": "memory"})
        self.memory = MemorySystem(self)
        from AGI_Evolutive.memory.semantic_manager import (  # type: ignore  # local import avoids circular init
            SemanticMemoryManager,
        )

        self.memory.semantic = SemanticMemoryManager(self.memory, architecture=self)

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
        self.language = SemanticUnderstanding(self, self.memory)

        self.concept_extractor = ConceptExtractor(data_dir="data")
        self.episodic_linker = EpisodicLinker(data_dir="data")

        self.action_interface = ActionInterface()
        self.perception_interface = PerceptionInterface()

        # Advanced subsystems
        self.style_profiler = StyleProfiler(persist_path="data/style_profiles.json")
        self.beliefs = BeliefGraph()
        self.user_model = UserModel()
        self.calibration = CalibrationMeter()
        self.novelty_detector = NoveltyDetector()
        self.abduction = AbductiveReasoner(self.beliefs, self.user_model)
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
        else:
            self.self_improver = None
            self.promotions = None

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

        surface = user_msg
        hints = {}
        try:
            hints = self.style_policy.adapt_from_instruction(user_msg)
        except Exception:
            hints = {}
        try:
            parsed = self.language.parse_utterance(user_msg, context={"style_hints": hints})
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            pass

        novelty_score = 0.0
        novelty_flag = False
        try:
            novelty_score, novelty_flag = self.novelty_detector.assess(surface, update=True)
            if novelty_flag and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="novelty_alert",
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
        except Exception:
            pass

        reason_out: Dict[str, Any] = {}
        if abduction_result:
            reason_out = dict(abduction_result.get("reason_out") or {})
        else:
            try:
                reason_out = self.reasoning.reason_about(surface, context={"inbox_docs": inbox_docs})
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
        response = format_agent_reply(base_text, **contract)

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
            ev_id = self.calibration.log_prediction(
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

    def _tick_background_systems(self) -> None:
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
            self.concept_extractor.step()
        except Exception:
            pass

        try:
            self.episodic_linker.step()
        except Exception:
            pass

        try:
            self.autonomy.tick()
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

    def _record_skill(self, concept: str) -> None:
        """Persiste un concept appris comme 'skill' + trace mémoire."""
        # trace mémoire
        try:
            self.memory.add_memory(kind='concept_learned', content=concept, metadata={'source':'user_confirm'})
        except Exception:
            pass
        # persistance sur disque
        import os, json, time
        skills_path = os.path.join('data', 'skills.json')
        try:
            os.makedirs('data', exist_ok=True)
            skills = {}
            if os.path.exists(skills_path):
                with open(skills_path, 'r', encoding='utf-8') as f:
                    skills = json.load(f)
            skills[str(concept)] = {
                'learned_at': time.time(),
                'source': 'user_confirm',
            }
            with open(skills_path, 'w', encoding='utf-8') as f:
                json.dump(skills, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # petit nudge côté Learning (si présent)
        try:
            if hasattr(self, 'learning') and hasattr(self.learning, 'learning_competencies'):
                v = self.learning.learning_competencies.get('skills_compiled', 0.0)
                self.learning.learning_competencies['skills_compiled'] = min(1.0, v + 0.01)
        except Exception:
            pass

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
