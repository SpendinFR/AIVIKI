from enum import Enum, auto
from typing import List, Dict, Any, Callable


class Stage(Enum):
    PERCEIVE = auto()
    ATTEND = auto()
    INTERPRET = auto()
    EVALUATE = auto()
    REFLECT = auto()
    REASON = auto()
    DECIDE = auto()
    ACT = auto()
    FEEDBACK = auto()
    LEARN = auto()
    UPDATE = auto()


class ActMode(Enum):
    REFLEX = auto()
    HABIT = auto()
    DELIBERATE = auto()


def _ACT(mode_or_fn):
    return {"stage": Stage.ACT, "mode": mode_or_fn}


REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "THREAT": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.REFLECT, "skip_if": lambda ctx: ctx["meta"].get("immediacy", 0.0) >= 0.8},
        {"stage": Stage.REASON, "skip_if": lambda ctx: ctx["meta"].get("immediacy", 0.0) >= 0.8},
        {"stage": Stage.DECIDE},
        _ACT(lambda ctx: ActMode.REFLEX if ctx["meta"].get("immediacy", 0.0) >= 0.8 else ActMode.DELIBERATE),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "GOAL": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.REFLECT},
        {"stage": Stage.REASON},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.DELIBERATE),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "CURIOSITY": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.REFLECT},
        {"stage": Stage.REASON},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.DELIBERATE),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "NEED": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "SIGNAL": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "HABIT": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "EMOTION": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.REFLECT},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "MEMORY_ASSOC": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.ATTEND},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
}


REGISTRY_UPDATE: Dict[str, List[Dict[str, Any]]] = {
    "SELF_JUDGMENT": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.EVALUATE},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.HABIT),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.LEARN},
        {"stage": Stage.UPDATE},
    ],
    "INTROSPECTION": [
        {"stage": Stage.PERCEIVE},
        {"stage": Stage.INTERPRET},
        {"stage": Stage.REFLECT},
        {"stage": Stage.REASON},
        {"stage": Stage.DECIDE},
        _ACT(ActMode.DELIBERATE),
        {"stage": Stage.FEEDBACK},
        {"stage": Stage.UPDATE},
    ],
}


REGISTRY.update(REGISTRY_UPDATE)
