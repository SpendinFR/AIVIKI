import time
from typing import Optional, Dict, Any

from typing import Any, Dict, List, Optional

# Import subsystems (existants)
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

from runtime.logger import JSONLLogger
from runtime.response import format_agent_reply, ensure_contract
from language.social_reward import extract_social_reward
from language.policy import StylePolicy
from goals.dag_store import GoalDAG
from autonomy.core import AutonomyCore
# Nouveaux modules (observabilité, autonomie, objectifs, style)
from autonomy.core import AutonomyCore
from goals.dag_store import GoalDAG
from language.policy import StylePolicy
from language.social_reward import extract_social_reward
from runtime.logger import JSONLLogger
from runtime.response import ensure_contract, format_agent_reply
# Telemetry
from core.telemetry import Telemetry


class CognitiveArchitecture:
    """
    Nœud central : instancie, relie les sous-systèmes, offre un cycle conversationnel,
    expose un statut cognitif, et lance l'autonomie idle.
    """

    def __init__(self):
        # Observabilité & politiques
        # Observabilité
        self.logger = JSONLLogger("runtime/agent_events.jsonl")
        self.style_policy = StylePolicy()
        self.goal_dag = GoalDAG("runtime/goal_dag.json")

        # Sous-systèmes
        # Instanciation des sous-systèmes
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
        self.creativity = CreativitySystem(self, self.memory, self.reasoning, self.emotions, self.metacognition)

        self.telemetry.log("init", "core", {"stage": "creativity"})
        self.creativity = CreativitySystem(
            self, self.memory, self.reasoning, self.emotions, self.metacognition
        )

        self.telemetry.log("init", "core", {"stage": "world_model"})
        self.world_model = PhysicsEngine(self, self.memory)

        self.telemetry.log("init", "core", {"stage": "language"})
        self.language = SemanticUnderstanding(self, self.memory)

        # Etat global
        self.global_activation = 0.5
        self.start_time = time.time()

    def cycle(self, user_msg: Optional[str]=None, inbox_docs=None):
        """
        Une boucle cognitive simple:
        - Percevoir / intégrer (si nécessaire)
        - Comprendre (Language v2)
        - Raisonner sommairement (si branché)
        - Répondre de façon introspective (self-ask si incertain)
        - (Optionnel) Apprendre / journaliser
        """
        response = None

        # Exemple d’intégration future: perception des inbox_docs
        # if inbox_docs: self.perception.ingest(inbox_docs)

        if user_msg:
            try:
                response = self.language.respond(user_msg, context={})
            except Exception:
                try:
                    parsed = self.language.parse_utterance(user_msg, context={})
                    response = f"Reçu: {parsed.surface_form if hasattr(parsed, 'surface_form') else user_msg}"
                except Exception:
                    response = f"Reçu: {user_msg}"
                # Language v2 -> réponse introspective
                response = self.language.respond(user_msg, context={
                    "global_activation": getattr(self, "global_activation", 0.5),
                })

                # Journalisation épisodique (si ton MemorySystem a une API ; sinon no-op)
                try:
                    if hasattr(self.memory, "store_interaction"):
                        self.memory.store_interaction({
                            "ts": time.time(),
                            "user": user_msg,
                            "agent": response,
                            "lang_state": getattr(self.language.state, "to_dict", lambda: {})(),
                        })
                except Exception:
                    pass

            except Exception:
                # fallback minimal
                response = f"Reçu: {user_msg}"
        else:
            # Pas de message utilisateur: on peut retourner un statut simple
            response = "OK"

        return response or "OK"
        # Autonomie idle
        self.autonomy = AutonomyCore(self, self.logger, self.goal_dag)
        self.autonomy.start()

        self.logger.write("system.init", ok=True, subsystems=list(self._present_subsystems().keys()))

    # ---- statut ----
    def _present_subsystems(self) -> Dict[str, bool]:
        names = ["memory", "perception", "reasoning", "goals", "metacognition", "emotions", "learning", "creativity", "world_model", "language"]
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
