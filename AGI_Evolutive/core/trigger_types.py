from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional


class TriggerType(Enum):
    NEED = auto()          # homeostasis (internal drives)
    GOAL = auto()          # explicit task/goal (incl. curiosity/meta-generated)
    CURIOSITY = auto()     # info gap not yet materialized as a goal
    THREAT = auto()        # danger/alert
    SIGNAL = auto()        # external interrupt / notification
    HABIT = auto()         # context-cued routine
    EMOTION = auto()       # affective peak drives priority
    MEMORY_ASSOC = auto()  # spontaneous recall/association


@dataclass
class Trigger:
    type: TriggerType
    # importance, probability, reversibility, effort, uncertainty, immediacy,
    # habit_strength, deadline_ts, source, etc.
    meta: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None  # raw content (user text, goal id, memory id, etc.)
