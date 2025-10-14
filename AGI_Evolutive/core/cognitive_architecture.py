# core/cognitive_architecture.py
import time
from typing import Optional

# Import subsystems
from memory import MemorySystem
from perception import PerceptionSystem
from reasoning import ReasoningSystem
from goals import GoalSystem
from emotions import EmotionalSystem
from learning import ExperientialLearning
from metacognition import MetacognitiveSystem
from creativity import CreativitySystem
from world_model import PhysicsEngine
from language import SemanticUnderstanding
from cognition.reward_engine import RewardEngine
from language.style_profiler import StyleProfiler


class CognitiveArchitecture:
    def __init__(self):
        # Instantiate core subsystems in dependency-safe order
        self.memory = MemorySystem(self)
        self.perception = PerceptionSystem(self, self.memory)
        self.reasoning = ReasoningSystem(self, self.memory, self.perception)
        self.goals = GoalSystem(self, self.memory, self.reasoning)
        self.metacognition = MetacognitiveSystem(self, self.memory, self.reasoning)
        self.emotions = EmotionalSystem(self, self.memory, self.metacognition)
        self.learning = ExperientialLearning(self)
        self.creativity = CreativitySystem(
            self, self.memory, self.reasoning, self.emotions, self.metacognition
        )
        self.world_model = PhysicsEngine(self, self.memory)
        self.language = SemanticUnderstanding(self, self.memory)

        # --- nouveaux sous-systèmes (Gap 1) ---
        self.style_profiler = StyleProfiler(persist_path="data/style_profiles.json")
        self.reward_engine = RewardEngine(
            architecture=self,
            memory=self.memory,
            emotions=self.emotions,
            goals=self.goals,
            metacognition=self.metacognition,
            persist_dir="data",
        )

        self.global_activation = 0.5
        self.start_time = time.time()
        self.last_output_text = ""
        self.last_user_id = "default"

    def cycle(self, user_msg: Optional[str] = None, inbox_docs=None, user_id: str = "default"):
        """
        Un cycle : perçoit -> (raisonne) -> répond, avec mimétisme stylistique et
        récompense extrinsèque dérivée du feedback utilisateur.
        """
        self.last_user_id = user_id or "default"

        if user_msg:
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

        response = None
        if user_msg:
            try:
                parsed = self.language.parse_utterance(user_msg, context={})
                response = f"Reçu: {parsed.surface_form if hasattr(parsed, 'surface_form') else user_msg}"
            except Exception:
                response = f"Reçu: {user_msg}"

        try:
            response = self.style_profiler.rewrite_to_match(response or "OK", self.last_user_id)
        except Exception:
            pass

        self.last_output_text = response or "OK"

        try:
            if self.memory and hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    "dialog_turn",
                    {
                        "t": time.time(),
                        "user_id": self.last_user_id,
                        "user_msg": user_msg,
                        "assistant_msg": self.last_output_text,
                    },
                )
        except Exception:
            pass

        return self.last_output_text

    # ----------------------------------------------------------------------
    # ✅ Méthode demandée par la métacognition : état global du système
    # ----------------------------------------------------------------------
    def get_cognitive_status(self) -> dict:
        """
        Retourne un résumé global de l'état cognitif actuel.
        Cette méthode est utilisée par la métacognition et la créativité.
        """
        status = {
            "global_activation": getattr(self, "global_activation", 0.5),
            "uptime_sec": round(time.time() - getattr(self, "start_time", time.time()), 2),
            "subsystems": {}
        }

        subsystems = [
            "memory", "reasoning", "goals", "emotions",
            "metacognition", "creativity", "learning"
        ]

        for name in subsystems:
            subsystem = getattr(self, name, None)
            if subsystem is None:
                continue

            # Cherche une méthode de statut dans le sous-système
            possible_getters = [
                "get_status",
                "get_state",
                "get_creative_status",
                "get_emotional_state",
                "get_learning_status"
            ]

            found = None
            for g in possible_getters:
                if hasattr(subsystem, g) and callable(getattr(subsystem, g)):
                    found = getattr(subsystem, g)
                    break

            if found:
                try:
                    status["subsystems"][name] = found()
                except Exception as e:
                    status["subsystems"][name] = {"error": str(e)}
            else:
                # Par défaut, inclut le type du sous-système
                status["subsystems"][name] = {"type": type(subsystem).__name__}

        return status
        # --- Robustesse vérité / longueur pendant l'init ---
    def __bool__(self) -> bool:
        # Toujours True pour éviter que bool(self) déclenche __len__ pendant l'init
        return True

    def __len__(self) -> int:
        # Ne JAMAIS accéder directement à des attributs potentiellement pas encore définis
        names = [
            "memory", "perception", "reasoning", "goals",
            "metacognition", "emotions", "learning",
            "creativity", "world_model", "language"
        ]
        count = 0
        for n in names:
            if getattr(self, n, None) is not None:
                count += 1
        return count


