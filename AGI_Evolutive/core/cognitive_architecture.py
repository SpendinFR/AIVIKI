import time
from typing import Any, Dict, List, Optional

# Import subsystems (existants)
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

# Nouveaux modules (observabilité, autonomie, objectifs, style)
from autonomy.core import AutonomyCore
from goals.dag_store import GoalDAG
from language.policy import StylePolicy
from language.social_reward import extract_social_reward
from runtime.logger import JSONLLogger
from runtime.response import ensure_contract, format_agent_reply


class CognitiveArchitecture:
    """
    Nœud central : instancie, relie les sous-systèmes, offre un cycle conversationnel,
    expose un statut cognitif, et lance l'autonomie idle.
    """

    def __init__(self):
        # Observabilité
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.style_policy = StylePolicy()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Instanciation des sous-systèmes
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

        # Etats globaux
        self.global_activation = 0.5
        self.start_time = time.time()

        # Autonomie (idle)
        self.autonomy = AutonomyCore(self, self.logger, self.goal_dag)
        self.autonomy.start()

        # Premier snapshot
        self.logger.write(
            "system.init", ok=True, subsystems=list(self._present_subsystems().keys())
        )

    # -------------- Observabilité / statut --------------
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
            "goal_focus": self.goal_dag.choose_next_goal(),
        }

    # -------------- Cycle conversationnel --------------
    def cycle(self, user_msg: Optional[str] = None, inbox_docs=None) -> str:
        """
        Un cycle lisible et tracé:
        - parse & hypothèses
        - choix + plan court
        - formate réponse (contrat)
        - met à jour reward social / style policy
        - logge tout
        """
        now = time.time()
        if user_msg is None:
            # no-op cycle (peut être appelé par le shell)
            return "OK"

        # Informe l'autonomie qu'il y a activité utilisateur
        try:
            self.autonomy.notify_user_activity()
        except Exception:
            pass

        # 1) Parse de l’énoncé
        try:
            parsed = self.language.parse_utterance(user_msg, context={})
            surface = getattr(parsed, "surface_form", user_msg)
        except Exception:
            surface = user_msg

        # 2) Génère 2–3 hypothèses d’intention simples (pour traçabilité)
        hypos: List[str] = [
            "tu veux une réponse non-générique avec plan actionnable",
            "tu veux que j’explique ce que j’apprends en temps réel",
            "tu veux que je propose un prochain test clair",
        ]

        # heuristique de choix
        chosen_idx = 0
        if "pourquoi" in surface.lower():
            chosen_idx = 1
        hypothese_choisie = hypos[chosen_idx]
        incertitude = 0.35 if chosen_idx in (0, 1, 2) else 0.5

        # 3) Prochain test selon style policy
        ask_more = self.style_policy.params.get("asking_rate", 0.4) > 0.35
        prochain_test = (
            "te proposer 2 options et valider la plus utile"
            if ask_more
            else "exécuter une mini-étape et te montrer le diff"
        )

        # 4) Base de texte (réponse de fond)
        base_text = self._generate_base_text(surface)

        # 5) Contrat de réponse
        contract = ensure_contract(
            {
                "hypothese_choisie": hypothese_choisie,
                "incertitude": incertitude,
                "prochain_test": prochain_test,
                "appris": [
                    "associer récompense sociale ↔ paramètres de style",
                    "tenir un journal d’épisodes de raisonnement",
                ],
                "besoin": [
                    "confirmer si tu préfères patchs de code immédiats ou d’abord schémas + tests"
                ],
            }
        )
        final = format_agent_reply(base_text, **contract)

        # 6) Récompense sociale (apprentissage)
        reward = extract_social_reward(user_msg).get("reward", 0.0)
        try:
            self.style_policy.update_from_reward(reward)
        except Exception:
            pass

        # 7) Log
        self.logger.write(
            "dialogue.turn",
            user_msg=user_msg,
            surface=surface,
            hypothesis=hypothese_choisie,
            incertitude=incertitude,
            test=prochain_test,
            reward=reward,
            style=self.style_policy.as_dict(),
        )

        # 8) Méta (optionnel & safe)
        try:
            if self.metacognition:
                self.metacognition._record_metacognitive_event(
                    event_type="dialogue_analysis",
                    domain=
                    self.metacognition.CognitiveDomain.LANGUAGE
                    if hasattr(self.metacognition, "CognitiveDomain")
                    else None,
                    description=f"Tour avec hypothèse '{hypothese_choisie}'",
                    significance=0.3,
                    confidence=1.0 - incertitude,
                )
        except Exception:
            pass

        return final

    # -------------- Génération “base” (sans LLM externe) --------------
    def _generate_base_text(self, surface: str) -> str:
        """
        Produit un cœur de réponse court, ancré dans l’état interne.
        (Tu peux plus tard router vers un générateur avancé.)
        """
        st = self.get_cognitive_status()
        status_line = (
            f"⏱️ {st['uptime_s']}s | 🔋 act={st['global_activation']:.2f} | 🧠 wm={st['working_memory_load']:.2f}"
        )
        focus = st["goal_focus"]
        focus_line = (
            f"🎯 focus: {focus['id']} (EVI={focus['evi']:.2f}, prog={focus['progress']:.2f})"
        )
        return f"Reçu: {surface}\n{status_line}\n{focus_line}"
