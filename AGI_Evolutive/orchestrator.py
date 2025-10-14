import time
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
