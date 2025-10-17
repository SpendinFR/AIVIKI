from AGI_Evolutive.core.trigger_types import Trigger, TriggerType


class TriggerRouter:
    def select_pipeline(self, t: Trigger) -> str:
        if t.type is TriggerType.THREAT:
            return "THREAT"
        if t.type is TriggerType.NEED:
            return "NEED"
        if t.type is TriggerType.GOAL:
            return "GOAL"
        if t.type is TriggerType.CURIOSITY:
            return "CURIOSITY"
        if t.type is TriggerType.SIGNAL:
            return "SIGNAL"
        if t.type is TriggerType.HABIT:
            return "HABIT"
        if t.type is TriggerType.EMOTION:
            return "EMOTION"
        if t.type is TriggerType.MEMORY_ASSOC:
            return "MEMORY_ASSOC"
        return t.type.name
