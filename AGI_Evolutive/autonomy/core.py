from __future__ import annotations

import threading
import time

from typing import Any, Dict, List

from AGI_Evolutive.goals.dag_store import GoalDAG
from AGI_Evolutive.reasoning.structures import (
    Evidence,
    Hypothesis,
    Test,
    episode_record,
)
from AGI_Evolutive.runtime.logger import JSONLLogger


class AutonomyCore:
    """
    Scheduler d'autonomie : en idle, choisit un sous-but à forte EVI,
    exécute UNE petite étape, logge, et propose un prochain test.
    """

    def __init__(self, arch, logger: JSONLLogger, dag: GoalDAG):
        self.arch = arch
        self.logger = logger
        self.dag = dag
        self.running = True
        self.thread = None
        self.idle_interval = 20  # secondes sans input pour tick
        self._last_user_time = time.time()
        self._tick = 0

    def notify_user_activity(self):
        self._last_user_time = time.time()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                idle_for = time.time() - self._last_user_time
                if idle_for >= self.idle_interval:
                    self.tick()
                    self._last_user_time = time.time()  # évite boucle frénétique
                time.sleep(1.0)
            except Exception as e:
                self.logger.write("autonomy.error", error=str(e))
                time.sleep(5)

    def tick(self):
        self._tick += 1
        # 1) Choix d'objectif
        pick = self.dag.choose_next_goal()
        goal_id, evi, progress = pick["id"], pick["evi"], pick["progress"]

        # 2) Hypothèse & test minimal
        h = [
            Hypothesis(
                content=f"Une micro-étape sur {goal_id} accélère la compréhension",
                prior=0.55,
            )
        ]
        t = [
            Test(
                description=f"Lire/agréger 3 traces récentes et distiller 1 règle pour {goal_id}",
                cost_est=0.15,
                expected_information_gain=min(0.2 + 0.5 * (1.0 - progress), 0.95),
            )
        ]

        # 3) "Exécution" symbolique (sans I/O lourde ici)
        rule = self._distill_micro_rule(goal_id)
        ev = Evidence(notes=f"Règle distillée: {rule}", confidence=0.6)

        # 3bis) Décider d'une action via la policy (si dispo)
        proposer = getattr(self.arch, "proposer", None)
        policy = getattr(self.arch, "policy", None)
        homeo = getattr(self.arch, "homeostasis", None) or getattr(self.arch, "homeo", None)
        planner = getattr(self.arch, "planner", None)
        memory = getattr(self.arch, "memory", None)
        proposals: List[Dict[str, Any]] = []
        if proposer and hasattr(proposer, "run_once_now"):
            try:
                raw_props = proposer.run_once_now() or []
                if isinstance(raw_props, list):
                    proposals = [p for p in raw_props if isinstance(p, dict)]
            except Exception as exc:
                self.logger.write("autonomy.warn", stage="proposer", error=str(exc))

        belief = 0.6
        try:
            self_model = getattr(self.arch, "self_model", None)
            if self_model and hasattr(self_model, "belief_confidence"):
                belief = float(self_model.belief_confidence({}))
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="belief_confidence", error=str(exc))

        novelty_fam = 0.7
        try:
            recent: List[Dict[str, Any]] = []
            if memory and hasattr(memory, "get_recent_memories"):
                raw_recent = memory.get_recent_memories(n=100)
                if isinstance(raw_recent, list):
                    recent = [m for m in raw_recent if isinstance(m, dict)]
            typs = [m.get("type", m.get("kind")) for m in recent]
            same = sum(1 for t in typs if t == "update")
            novelty_fam = max(0.2, min(0.95, 1.0 - (0.02 * max(0, 5 - same))))
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="novelty_eval", error=str(exc))

        decision: Dict[str, Any] = {"decision": "noop", "reason": "no policy", "confidence": 0.5}
        if policy and hasattr(policy, "decide"):
            try:
                decision = policy.decide(
                    proposals,
                    self_state={"tick": self._tick},
                    proposer=proposer,
                    homeo=homeo,
                    planner=planner,
                    ctx={"belief_confidence": belief, "novelty_familiarity": novelty_fam},
                )
            except Exception as exc:
                decision = {"decision": "error", "reason": str(exc), "confidence": 0.3}
                self.logger.write("autonomy.warn", stage="policy_decide", error=str(exc))

        # 4) Mise à jour DAG + logs
        progress_after = self.dag.bump_progress(0.01)
        ep = episode_record(
            user_msg="[idle]",
            hypotheses=h,
            chosen_index=0,
            tests=t,
            evidence=ev,
            result_text=f"Micro-étape sur {goal_id}: {rule}",
            final_confidence=0.62,
        )
        self.logger.write(
            "autonomy.tick",
            goal=goal_id,
            evi=evi,
            progress_before=progress,
            progress_after=progress_after,
            episode=ep,
            policy_decision=decision,
        )

        # 5) Ping métacognition (si existante)
        try:
            if hasattr(self.arch, "metacognition") and self.arch.metacognition:
                self.arch.metacognition._record_metacognitive_event(
                    event_type="autonomy_step",
                    domain=
                    self.arch.metacognition.CognitiveDomain.LEARNING
                    if hasattr(self.arch.metacognition, "CognitiveDomain")
                    else None,
                    description=f"Idle→micro-étape sur {goal_id}",
                    significance=min(0.3 + 0.4 * evi, 0.9),
                    confidence=0.6,
                )
        except Exception:
            pass

        # 6) Feedback à la policy
        try:
            executed = decision.get("proposal") if isinstance(decision, dict) else None
            success = bool(decision.get("decision") == "apply") if isinstance(decision, dict) else False
            if policy and hasattr(policy, "register_outcome") and executed:
                policy.register_outcome(executed, success)
        except Exception as exc:
            self.logger.write("autonomy.warn", stage="policy_feedback", error=str(exc))

    def _distill_micro_rule(self, goal_id: str) -> str:
        """
        Distille une mini-règle depuis l'historique récent pour garder l'agent 'vivant'.
        (Heuristique très simple et sûre.)
        """
        try:
            # lit les 30 derniers événements dialogue/autonomy (si dispo via logger → non trivial)
            # ici : renvoie une règle statique contextualisée pour démarrer
            if goal_id == "understand_humans":
                return "Toujours expliciter l'hypothèse et demander 1 validation binaire."
            if goal_id == "self_modeling":
                return "Journaliser 'ce que j'ai appris' au moins une fois par tour."
            if goal_id == "tooling_mastery":
                return "Proposer un patch minimal plutôt qu'un grand refactor."
            return "Faire un pas plus petit mais mesurable."
        except Exception:
            return "Faire un pas plus petit mais mesurable."
