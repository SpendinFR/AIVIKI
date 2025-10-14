import time
from core.config import load_config
from core.self_model import SelfModel
from core.policy import PolicyEngine
from memory.memory_store import MemoryStore
from memory.consolidator import Consolidator
from memory.concept_extractor import ConceptExtractor
from memory.episodic_linker import EpisodicLinker
from cognition.homeostasis import Homeostasis
from cognition.proposer import Proposer
from cognition.planner import Planner
from cognition.meta_cognition import MetaCognition
from cognition.reflection_loop import ReflectionLoop
from cognition.emotion_engine import EmotionEngine
from cognition.evolution_manager import EvolutionManager
from io.perception_interface import PerceptionInterface
from io.action_interface import ActionInterface
from scheduler import Scheduler

class Orchestrator:
    """
    Cycle de vie:
    perceive -> memorize -> consolidate -> concept/episodic -> emotion -> homeostasis
    -> meta-cognition (incertitude->learning goals) -> planning -> action -> proposals
    -> evolution manager (macro)
    """
    def __init__(self, arch):
        load_config()
        self.arch = arch
        # Bases
        self.self_model = SelfModel()
        self.policy = PolicyEngine()
        self.memory = MemoryStore()
        # Cognition / Mémoire élargies
        self.homeo = Homeostasis()
        self.planner = Planner()
        self.consolidator = Consolidator(self.memory)
        self.concepts = ConceptExtractor(self.memory)
        self.epi = EpisodicLinker(self.memory)
        self.meta = MetaCognition(self.memory, self.planner, self.self_model)
        self.emotions = EmotionEngine()
        self.proposer = Proposer(self.memory, self.planner, self.homeo)
        self.actions = ActionInterface(self.memory)
        self.perception = PerceptionInterface(self.memory)
        self.evolution = EvolutionManager()
        # Réflexion périodique (inner monologue) – désactivable si tu préfères un scheduler pur
        self.reflect_loop = ReflectionLoop(self.meta, interval_sec=300)
        self.reflect_loop.start()
        # Scheduler (sans threads nécessaires)
        self.scheduler = Scheduler()
        self._register_jobs()

        # Boot log
        self.memory.add_memory({"kind":"system","text":"Orchestrator initialized","ts":time.time()})

    def _register_jobs(self):
        self.scheduler.register_job("scan_inbox", 30, lambda: self.perception.scan_inbox())
        self.scheduler.register_job("concepts", 180, lambda: self.concepts.extract_from_recent(200))
        self.scheduler.register_job("episodic_links", 120, lambda: self.epi.link_recent(80))

    # --------- Cycles élémentaires ---------
    def observe(self, user_msg: str = None):
        if user_msg:
            self.perception.ingest_user_utterance(user_msg, author="user")
        else:
            self.memory.add_memory({"kind":"tick","text":"idle cycle","ts":time.time()})

    def consolidate(self):
        result = self.consolidator.run_once_now()
        if result["lessons"]:
            self.memory.add_memory({"kind":"lesson","text":" | ".join(result["lessons"]),"ts":time.time()})

    def emotion_homeostasis_cycle(self):
        recent = self.memory.get_recent_memories(n=60)
        self.emotions.update_from_recent_memories(recent)
        self.emotions.modulate_homeostasis(self.homeo)
        # intr/extr rewards
        r_intr = self.homeo.compute_intrinsic_reward(info_gain=0.5, progress=0.5)
        r_extr = self.homeo.compute_extrinsic_reward_from_memories("")  # plug feedback text si dispo
        return r_intr, r_extr

    def meta_cycle(self):
        # incertitude -> learning goals (créés automatiquement)
        assessment = self.meta.assess_understanding()
        goals = self.meta.propose_learning_goals(max_goals=2)
        return assessment, goals

    def planning_cycle(self):
        # garantir un grand objectif structurel
        self.planner.plan_for_goal("understand_humans", "Comprendre les humains")
        if not self.planner.state["plans"]["understand_humans"]["steps"]:
            self.planner.add_step("understand_humans","Observer un échange et extraire intentions")
            self.planner.add_step("understand_humans","Tester une hypothèse d’intention par question")

    def action_cycle(self):
        # traiter d’abord un goal d’apprentissage si présent
        picked = None
        for gid, p in self.planner.state["plans"].items():
            # priorité aux goals learn_*
            if gid.startswith("learn_"):
                picked = gid; break
        if not picked:
            picked = "understand_humans"

        step = self.planner.pop_next_action(picked)
        if not step: return
        # mapping simple desc->action
        desc = step["desc"].lower()
        action = {"type":"simulate","desc":desc}
        if "poser" in desc or "question" in desc:
            action = {"type":"communicate","text":"Peux-tu me décrire ton émotion actuelle et pourquoi ?", "target":"human"}
        elif "observer" in desc:
            action = {"type":"simulate","what":"observe_exchange"}

        res = self.actions.execute(action)
        self.planner.mark_action_done(picked, step["id"], success=bool(res.get("ok", True)))

    def proposals_cycle(self):
        props = self.proposer.run_once_now()
        for p in props:
            try:
                self.self_model.apply_proposal(p, self.policy)
            except Exception as e:
                self.memory.add_memory({"kind":"error","text":f"Proposal error: {e}","ts":time.time()})

    # --------- Tick public ---------
    def run_once_cycle(self, user_msg: str = None):
        # scheduler (scan inbox / concepts / episodic links)
        self.scheduler.tick()

        # Percevoir
        self.observe(user_msg)

        # Consolider & extraire structure
        self.consolidate()

        # Émotion & homeostasis
        r_intr, r_extr = self.emotion_homeostasis_cycle()

        # Métacognition (incertitude->goals)
        assessment, _ = self.meta_cycle()

        # Planifier & agir
        self.planning_cycle()
        self.action_cycle()

        # Propositions d’évolution
        self.proposals_cycle()

        # Journal macro pour évolution
        learning_rate = 0.5  # si tu as une vraie mesure, remplace ici
        self.evolution.log_cycle(intrinsic=r_intr, extrinsic=r_extr, learning_rate=learning_rate, uncertainty=assessment["uncertainty"])

        # (optionnel) Tous les 20 cycles, proposer des ajustements macro
        if self.evolution.state["cycle_count"] % 20 == 0:
            notes = self.evolution.propose_macro_adjustments()
            if notes:
                self.memory.add_memory({"kind":"strategy","text":" | ".join(notes),"ts":time.time()})
from typing import Optional

from core.config import load_config
from core.policy import PolicyEngine
from core.self_model import SelfModel
from memory.memory_store import MemoryStore
from memory.consolidator import Consolidator
from cognition.homeostasis import Homeostasis
from cognition.proposer import Proposer
from cognition.planner import Planner


class Orchestrator:
    """Coordinate high-level cycles across cognitive subsystems."""

    def __init__(self, arch) -> None:
        load_config()
        self.arch = arch
        self.self_model = SelfModel()
        self.policy = PolicyEngine()
        self.memory = MemoryStore()
        self.homeo = Homeostasis()
        self.planner = Planner()
        self.consolidator = Consolidator(self.memory)
        self.proposer = Proposer(self.memory, self.planner, self.homeo)

    def observe_and_memorize(self, user_msg: Optional[str] = None) -> None:
        payload = {
            "kind": "interaction" if user_msg else "tick",
            "text": user_msg or "(idle)",
            "ts": time.time(),
        }
        self.memory.add_memory(payload)

    def consolidate(self) -> None:
        result = self.consolidator.run_once_now()
        if result["lessons"]:
            self.memory.add_memory({"kind": "lesson", "text": " | ".join(result["lessons"]), "ts": time.time()})
        for proposal in result["proposals"]:
            try:
                self.self_model.apply_proposal(proposal, self.policy)
            except Exception:
                pass

    def homeostasis_cycle(self) -> None:
        info_gain = 0.5
        progress = 0.5
        intrinsic = self.homeo.compute_intrinsic_reward(info_gain, progress)
        extrinsic = self.homeo.compute_extrinsic_reward_from_memories("")
        self.homeo.update_from_rewards(intrinsic, extrinsic)

    def planning_cycle(self) -> None:
        self.planner.plan_for_goal("understand_humans", "Comprendre les humains")
        plan = self.planner.state["plans"]["understand_humans"]
        if not plan["steps"]:
            self.planner.add_step("understand_humans", "Observer un échange et extraire intentions")
            self.planner.add_step("understand_humans", "Tester une hypothèse d’intention par question ciblée")

    def act_or_simulate(self) -> None:
        step = self.planner.pop_next_action("understand_humans")
        if not step:
            return
        sim = self.planner.simulate_action(step["desc"])
        self.memory.add_memory({"kind": "action_sim", "text": step["desc"], "sim": sim, "ts": time.time()})
        self.planner.mark_action_done("understand_humans", step["id"], success=True)

    def propose_and_apply(self) -> None:
        proposals = self.proposer.run_once_now()
        for proposal in proposals:
            try:
                self.self_model.apply_proposal(proposal, self.policy)
            except Exception:
                pass

    def run_once_cycle(self, user_msg: Optional[str] = None) -> None:
        self.observe_and_memorize(user_msg)
        self.consolidate()
        self.homeostasis_cycle()
        self.planning_cycle()
        self.act_or_simulate()
        self.propose_and_apply()
