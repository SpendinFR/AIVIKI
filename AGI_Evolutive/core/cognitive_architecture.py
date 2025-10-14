# core/cognitive_architecture.py
import time
from typing import Optional

# Subsystems
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

# Telemetry
from core.telemetry import Telemetry


class CognitiveArchitecture:
    def __init__(self):
        self.telemetry = Telemetry()
        self.global_activation = 0.5
        self.start_time = time.time()
        self.reflective_mode = True

        # Instanciation des sous-systèmes avec télémétrie détaillée
        self.telemetry.log("init", "core", {"stage": "memory"})
        self.memory = MemorySystem(self)

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

        self.telemetry.log("ready", "core", {"status": "initialized"})

    # ===================== OUTILS DIAGNOSTIC =====================

    def get_cognitive_status(self) -> dict:
        """
        Statut global lisible (et robuste). Utilisé par méta/ressources/CLI.
        """

        def safe_len(x):
            try:
                return len(x)
            except Exception:
                return 0

        # Reasoning
        r = getattr(self, "reasoning", None)
        r_hist = getattr(r, "reasoning_history", {}) if r else {}
        recent_inf = list(r_hist.get("recent_inferences", []))[-5:] if isinstance(r_hist, dict) else []
        avg_conf = getattr(r, "get_reasoning_stats", lambda: {"average_confidence": 0.5})()
        if isinstance(avg_conf, dict):
            avg_conf = avg_conf.get("average_confidence", 0.5)

        # Creativity
        c = getattr(self, "creativity", None)
        ideas = []
        try:
            pool = c.idea_generation.get("idea_pool", []) if c else []
            ideas = list(pool)[-10:] if pool else []
        except Exception:
            pass

        # Emotions
        e = getattr(self, "emotions", None)
        mood = getattr(e, "current_mood", "neutral")

        # Metacognition
        m = getattr(self, "metacognition", None)
        meta_events = 0
        try:
            meta_events = len(m.metacognitive_history["events"]) if m else 0
        except Exception:
            pass

        # Memory
        mem = getattr(self, "memory", None)
        mem_size = 0
        try:
            mem_size = getattr(mem, "size", lambda: 0)()
        except Exception:
            pass

        # Telemetry snapshot
        telemetry = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {}

        return {
            "uptime_sec": int(time.time() - self.start_time),
            "global_activation": float(getattr(self, "global_activation", 0.5)),
            "reasoning": {
                "recent_inferences": safe_len(recent_inf),
                "avg_confidence": float(avg_conf) if isinstance(avg_conf, (int, float)) else 0.5,
            },
            "creativity": {
                "recent_ideas": safe_len(ideas),
            },
            "emotions": {
                "mood": mood,
            },
            "metacognition": {
                "events": meta_events,
            },
            "memory": {
                "approx_size": mem_size,
            },
            "telemetry": telemetry,
        }

    def diagnostic_snapshot(self, tail=30) -> dict:
        """Renvoie un snapshot: statut + derniers événements télémétrie."""
        return {
            "status": self.get_cognitive_status(),
            "tail": self.telemetry.tail(tail),
        }

    def toggle_reflective_mode(self, on: bool):
        self.reflective_mode = bool(on)
        self.telemetry.log("cfg", "core", {"reflective_mode": self.reflective_mode})

    # ===================== BOUCLE PRINCIPALE =====================

    def cycle(self, user_msg: Optional[str] = None, inbox_docs=None):
        """Cycle cognitif simple."""
        response = None
        if user_msg:
            self.telemetry.log("input", "language", {"text": user_msg})
            if self.reflective_mode and hasattr(self.language, "generate_reflective_reply"):
                try:
                    response = self.language.generate_reflective_reply(self, user_msg)
                    self.telemetry.log("output", "language", {"mode": "reflective"})
                except Exception as exc:
                    self.telemetry.log("error", "language", {"where": "generate_reflective_reply", "err": str(exc)})
            if not response:
                try:
                    parsed = self.language.parse_utterance(user_msg, context={})
                    surface = parsed.surface_form if hasattr(parsed, "surface_form") else user_msg
                    response = f"Reçu: {surface}"
                except Exception:
                    response = f"Reçu: {user_msg}"
        return response or "OK"
