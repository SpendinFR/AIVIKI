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
from cognition.evolution_manager import EvolutionManager
from runtime.scheduler import Scheduler


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
        self.global_activation = 0.5
        self.start_time = time.time()

        # === Évolution long-terme ===
        self.evolution = EvolutionManager(data_dir="data")
        self.evolution.bind(
            architecture=self,
            memory=getattr(self, "memory", None),
            metacog=getattr(self, "metacognition", None),
            goals=getattr(self, "goals", None),
            learning=getattr(self, "learning", None),
            emotions=getattr(self, "emotions", None),
            language=getattr(self, "language", None),
        )

        # === Scheduler (orchestration en arrière-plan) ===
        self.scheduler = Scheduler(self, data_dir="data")
        self.scheduler.start()  # thread daemon, non bloquant

    def cycle(self, user_msg: Optional[str] = None, inbox_docs=None):
        """One simple cognitive cycle: perceive -> reason -> plan -> act -> learn -> reflect."""
        response = None
        if user_msg:
            # Use language understanding to parse, then use generation to reply
            try:
                parsed = self.language.parse_utterance(user_msg, context={})
                response = f"Reçu: {parsed.surface_form if hasattr(parsed, 'surface_form') else user_msg}"
            except Exception:
                response = f"Reçu: {user_msg}"

        # Après le cycle court, on peut pousser un enregistrement d'état long-terme à faible cadence
        try:
            if hasattr(self, "evolution") and self.evolution:
                # léger throttling interne dans EvolutionManager si besoin (au pire c'est O(1) ici)
                self.evolution.record_cycle(extra_tags={"via": "cycle"})
        except Exception:
            pass

        # idem concept/épisode si pas déjà déclenchés : (facultatif, le scheduler s'en charge)
        try:
            if hasattr(self, "concept_extractor") and self.concept_extractor:
                self.concept_extractor.step()
        except Exception:
            pass
        try:
            if hasattr(self, "episodic_linker") and self.episodic_linker:
                self.episodic_linker.step()
        except Exception:
            pass
        return response or "OK"

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


