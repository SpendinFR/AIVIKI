import time
from typing import Any, Dict, Optional

from autonomy import AutonomyManager
from cognition.evolution_manager import EvolutionManager
from cognition.reward_engine import RewardEngine
from core.telemetry import Telemetry
from creativity import CreativitySystem
from emotions import EmotionalSystem
from goals import GoalSystem
from goals.dag_store import GoalDAG
from io.action_interface import ActionInterface
from io.perception_interface import PerceptionInterface
from language import SemanticUnderstanding
from language.policy import StylePolicy
from language.social_reward import extract_social_reward
from language.style_profiler import StyleProfiler
from learning import ExperientialLearning
from memory import MemorySystem
from memory.concept_extractor import ConceptExtractor
from memory.episodic_linker import EpisodicLinker
from metacognition import MetacognitiveSystem
from perception import PerceptionSystem
from reasoning import ReasoningSystem
from runtime.logger import JSONLLogger
from runtime.response import ensure_contract, format_agent_reply
from runtime.scheduler import Scheduler
from world_model import PhysicsEngine


class CognitiveArchitecture:
    """Central coordinator for the agent's cognitive subsystems."""

    def __init__(self):
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
        from memory.semantic_manager import (  # type: ignore  # local import avoids circular init
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

        surface = user_msg
        try:
            parsed = self.language.parse_utterance(user_msg, context={"style_hints": {}})
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            pass

        reason_out: Dict[str, Any] = {}
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

        apprentissages = [
            "associer récompense sociale ↔ style",
            "tenir un journal d'épisodes de raisonnement",
        ] + list(reason_out.get("appris", []))

        contract = ensure_contract(
            {
                "hypothese_choisie": reason_out.get("chosen_hypothesis", "clarifier intention"),
                "incertitude": float(
                    max(0.0, min(1.0, 1.0 - float(reason_out.get("final_confidence", 0.5))))
                ),
                "prochain_test": reason_out.get("prochain_test") or "-",
                "appris": apprentissages,
                "besoin": ["confirmer si tu veux patch immédiat ou plan en étapes"],
            }
        )

        base_text = self._generate_base_text(surface, reason_out)
        response = format_agent_reply(base_text, **contract)

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

        self._tick_background_systems()
        return response

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
