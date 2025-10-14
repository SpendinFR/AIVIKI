# core/cognitive_architecture.py
import time
from typing import Optional, Dict, Any

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

from runtime.logger import JSONLLogger
from runtime.response import format_agent_reply, ensure_contract
from language.social_reward import extract_social_reward
from language.policy import StylePolicy
from goals.dag_store import GoalDAG
from autonomy.core import AutonomyCore


class CognitiveArchitecture:
    def __init__(self):
        # Observabilité & politiques
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.style_policy = StylePolicy()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Sous-systèmes
        self.memory = MemorySystem(self)
        self.perception = PerceptionSystem(self, self.memory)
        self.reasoning = ReasoningSystem(self, self.memory, self.perception)
        self.goals = GoalSystem(self, self.memory, self.reasoning)
        self.metacognition = MetacognitiveSystem(self, self.memory, self.reasoning)
        self.emotions = EmotionalSystem(self, self.memory, self.metacognition)
        self.learning = ExperientialLearning(self)
        self.creativity = CreativitySystem(self, self.memory, self.reasoning, self.emotions, self.metacognition)
        self.world_model = PhysicsEngine(self, self.memory)
        self.language = SemanticUnderstanding(self, self.memory)

        # Etat global
        self.global_activation = 0.5
        self.start_time = time.time()

        # Autonomie idle
        self.autonomy = AutonomyCore(self, self.logger, self.goal_dag)
        self.autonomy.start()

        self.logger.write("system.init", ok=True, subsystems=list(self._present_subsystems().keys()))

    # ---- statut ----
    def _present_subsystems(self) -> Dict[str, bool]:
        names = ["memory", "perception", "reasoning", "goals", "metacognition", "emotions", "learning", "creativity", "world_model", "language"]
        return {n: hasattr(self, n) and getattr(self, n) is not None for n in names}

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
            "goal_focus": self.goal_dag.choose_next_goal()
        }

    # ---- cycle dialogue ----
    def cycle(self, user_msg: Optional[str] = None, inbox_docs=None) -> str:
        if user_msg is None:
            return "OK"

        # Reset idle
        try:
            self.autonomy.notify_user_activity()
        except Exception:
            pass

        # Parse
        try:
            parsed = self.language.parse_utterance(user_msg, context={})
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            surface = user_msg

        # Raisonner vraiment
        reason_out = {}
        try:
            reason_out = self.reasoning.reason_about(surface, context={"inbox_docs": inbox_docs})
        except Exception as e:
            # fallback: garde une trace d’erreur mais ne casse pas la réponse
            self.logger.write("reasoning.error", error=str(e), user_msg=surface)
            reason_out = {
                "summary": "Raisonnement basique uniquement (fallback).",
                "chosen_hypothesis": "clarifier intention + proposer 1 test",
                "tests": ["proposer 2 options et valider"],
                "final_confidence": 0.5,
                "appris": ["garder une trace même en cas d’erreur"],
                "prochain_test": "valider l’option la plus utile",
                "episode": None
            }

        # Contrat de réponse enrichi par le raisonnement
        apprentissages = [
            "associer récompense sociale ↔ style",
            "tenir un journal d’épisodes de raisonnement"
        ] + (reason_out.get("appris") or [])

        contract = ensure_contract({
            "hypothese_choisie": reason_out.get("chosen_hypothesis", "clarifier intention"),
            "incertitude": float(max(0.0, min(1.0, 1.0 - float(reason_out.get("final_confidence", 0.5))))),
            "prochain_test": (reason_out.get("prochain_test") or "—"),
            "appris": apprentissages,
            "besoin": ["confirmer si tu veux patch immédiat ou plan en étapes"]
        })

        base_text = self._generate_base_text(surface, reason_out)

        final = format_agent_reply(base_text, **contract)

        # Reward social → adapter style
        reward = extract_social_reward(user_msg).get("reward", 0.0)
        try:
            self.style_policy.update_from_reward(reward)
        except Exception:
            pass

        # Logs
        self.logger.write(
            "dialogue.turn",
            user_msg=user_msg,
            surface=surface,
            hypothesis=contract["hypothese_choisie"],
            incertitude=contract["incertitude"],
            test=contract["prochain_test"],
            reward=reward,
            style=self.style_policy.as_dict(),
            reason_summary=reason_out.get("summary", "")
        )

        # Méta signal
        try:
            from metacognition import CognitiveDomain
            if self.metacognition:
                self.metacognition._record_metacognitive_event(
                    event_type="dialogue_analysis",
                    domain=CognitiveDomain.LANGUAGE,
                    description=f"Hypothèse '{contract['hypothese_choisie']}'",
                    significance=0.35,
                    confidence=float(reason_out.get("final_confidence", 0.5))
                )
        except Exception:
            pass

        return final

    def _generate_base_text(self, surface: str, reason_out: Dict[str, Any]) -> str:
        st = self.get_cognitive_status()
        status_line = f"⏱️{st['uptime_s']}s | 🔋act={st['global_activation']:.2f} | 🧠wm={st['working_memory_load']:.2f}"
        focus = st["goal_focus"]
        focus_line = f"🎯focus:{focus['id']} (EVI={focus['evi']:.2f}, prog={focus['progress']:.2f})"
        rsum = reason_out.get("summary", "")
        return f"Reçu: {surface}\n{status_line}\n{focus_line}\n🧠 {rsum}"
