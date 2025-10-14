from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import re, time, json, os


@dataclass
class RewardEvent:
    timestamp: float
    user_id: str
    text: str
    channel: str               # "chat", "system", "tool"
    extrinsic_reward: float    # [-1.0, 1.0]
    polarity: str              # "positive" | "negative" | "neutral"
    intensity: float           # [0, 1] (force du signal)
    features: Dict[str, Any]   # détails (emojis, !, lexiques, etc.)
    context: Dict[str, Any]    # ex: last_assistant_output, active_goal_id, emotional_state


class RewardEngine:
    """
    Analyse le feedback social/utilisateur et génère des récompenses extrinsèques
    + ajuste drives/émotions/goals + log en mémoire + notifie métacognition.
    """

    POS_FR = {
        "bravo",
        "bien",
        "super",
        "parfait",
        "excellent",
        "incroyable",
        "merci",
        "génial",
        "top",
        "nickel",
        "parfait",
        "cool",
        "magnifique",
        "impeccable",
    }
    NEG_FR = {
        "nul",
        "mauvais",
        "horrible",
        "pas bien",
        "déçu",
        "décevant",
        "non",
        "faux",
        "pourri",
        "n'importe quoi",
        "t'es nul",
        "nulle",
        "c'est nul",
    }
    POS_EN = {
        "great",
        "good",
        "nice",
        "awesome",
        "amazing",
        "perfect",
        "thanks",
        "thank you",
        "well done",
        "brilliant",
        "wonderful",
        "excellent",
    }
    NEG_EN = {
        "bad",
        "terrible",
        "awful",
        "wrong",
        "no",
        "useless",
        "garbage",
        "disappointing",
        "nonsense",
        "makes no sense",
        "stupid",
        "dumb",
    }
    EMOJI_POS = set(list("👍👌👏💯✨🔥😄😁😊🤝❤️💪🤩🙌🌟✅🆗🙂🤗"))
    EMOJI_NEG = set(list("👎😡😠😞😔💢❌🛑🤬😤🙄😒"))

    def __init__(
        self,
        architecture=None,
        memory=None,
        emotions=None,
        goals=None,
        metacognition=None,
        persist_dir="data",
    ):
        self.arch = architecture
        self.memory = memory
        self.emotions = emotions
        self.goals = goals
        self.metacognition = metacognition
        self.persist_dir = persist_dir
        os.makedirs(os.path.join(self.persist_dir, "logs"), exist_ok=True)

    def ingest_user_message(
        self,
        user_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        channel: str = "chat",
    ) -> RewardEvent:
        """
        À appeler depuis la boucle de dialogue - analyse textuelle, calcule une récompense,
        met à jour systèmes, et retourne l'événement.
        """
        ev = self._analyze_feedback(user_id, text, context or {}, channel)
        self._apply_reward(ev)
        self._log_reward_event(ev)
        self._notify_metacognition(ev)
        return ev

    def _analyze_feedback(
        self, user_id: str, text: str, context: Dict[str, Any], channel: str
    ) -> RewardEvent:
        t = text.strip()
        t_lower = t.lower()

        pos_hits = self._count_hits(t_lower, self.POS_FR) + self._count_hits(
            t_lower, self.POS_EN
        )
        neg_hits = self._count_hits(t_lower, self.NEG_FR) + self._count_hits(
            t_lower, self.NEG_EN
        )

        emojis = [ch for ch in t if ch in self.EMOJI_POS or ch in self.EMOJI_NEG]
        pos_emoji = sum(1 for e in emojis if e in self.EMOJI_POS)
        neg_emoji = sum(1 for e in emojis if e in self.EMOJI_NEG)

        exclam = t.count("!")
        quest = t.count("?")
        caps_ratio = self._caps_ratio(t)

        direct_pos = 1 if re.search(r"\b(merci|thanks)\b", t_lower) else 0
        direct_neg = (
            1
            if re.search(
                r"\b(ta\sfaux|you(\sare)?\swrong|c'?est\s(faux|nul))\b", t_lower
            )
            else 0
        )

        raw_pos = pos_hits + pos_emoji + direct_pos
        raw_neg = neg_hits + neg_emoji + direct_neg
        raw = raw_pos - raw_neg

        base = 0.0
        if raw > 0:
            base = min(1.0, 0.2 + 0.15 * raw)
        elif raw < 0:
            base = max(-1.0, -0.2 + 0.15 * raw)

        intensity = min(1.0, 0.3 + 0.1 * exclam + 0.05 * quest + 0.6 * caps_ratio)
        extrinsic = float(max(-1.0, min(1.0, base * (0.6 + 0.4 * intensity))))

        polarity = "neutral"
        if extrinsic > 0.05:
            polarity = "positive"
        elif extrinsic < -0.05:
            polarity = "negative"

        features = {
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "pos_emoji": pos_emoji,
            "neg_emoji": neg_emoji,
            "exclam": exclam,
            "question": quest,
            "caps_ratio": caps_ratio,
            "intensity": intensity,
            "emoji_list": emojis,
        }
        return RewardEvent(
            timestamp=time.time(),
            user_id=user_id,
            text=text,
            channel=channel,
            extrinsic_reward=extrinsic,
            polarity=polarity,
            intensity=intensity,
            features=features,
            context=context,
        )

    def _apply_reward(self, ev: RewardEvent):
        try:
            emo = self.emotions
            if emo:
                if hasattr(emo, "register_social_feedback"):
                    emo.register_social_feedback(
                        ev.extrinsic_reward,
                        ev.intensity,
                        ev.polarity,
                        meta=asdict(ev),
                    )
                else:
                    if hasattr(emo, "state"):
                        st = emo.state
                        st["valence"] = max(
                            -1.0,
                            min(1.0, st.get("valence", 0.0) + 0.3 * ev.extrinsic_reward),
                        )
                        st["arousal"] = max(
                            0.0,
                            min(1.0, st.get("arousal", 0.5) + 0.2 * ev.intensity),
                        )
        except Exception:
            pass

        try:
            if self.goals:
                active = None
                if hasattr(self.goals, "get_active_goal"):
                    active = self.goals.get_active_goal()
                elif hasattr(self.goals, "current_goal"):
                    active = getattr(self.goals, "current_goal")

                if active:
                    lr = 0.25 * (0.5 + 0.5 * ev.intensity)
                    new_val = max(
                        0.0,
                        min(1.0, active.get("value", 0.5) + lr * ev.extrinsic_reward),
                    )
                    active["value"] = new_val
                    if "evidence" not in active:
                        active["evidence"] = []
                    active["evidence"].append(
                        {
                            "t": ev.timestamp,
                            "type": "social_feedback",
                            "delta_value": lr * ev.extrinsic_reward,
                            "user_id": ev.user_id,
                            "text": ev.text,
                        }
                    )
                    if hasattr(self.goals, "update_goal"):
                        self.goals.update_goal(active.get("id"), active)
        except Exception:
            pass

        try:
            if self.memory:
                payload = {
                    "type": "social_feedback",
                    "reward": ev.extrinsic_reward,
                    "polarity": ev.polarity,
                    "intensity": ev.intensity,
                    "features": ev.features,
                    "user_id": ev.user_id,
                    "text": ev.text,
                    "context": ev.context,
                }
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory("feedback", payload)
                elif hasattr(self.memory, "store_event"):
                    self.memory.store_event("feedback", payload)
        except Exception:
            pass

    def _notify_metacognition(self, ev: RewardEvent):
        try:
            m = self.metacognition
            if not m:
                return
            if hasattr(m, "_record_metacognitive_event"):
                m._record_metacognitive_event(
                    event_type="social_feedback",
                    domain=getattr(m, "CognitiveDomain", None).DECISION_MAKING
                    if hasattr(m, "CognitiveDomain")
                    else None,
                    description=f"Feedback {ev.polarity} (r={ev.extrinsic_reward:.2f}, I={ev.intensity:.2f})",
                    significance=abs(ev.extrinsic_reward) * ev.intensity,
                    confidence=0.7,
                    emotional_valence=ev.extrinsic_reward,
                    cognitive_load=0.0,
                    related_memories=[],
                    action_taken=None,
                )
        except Exception:
            pass

    def _log_reward_event(self, ev: RewardEvent):
        try:
            path = os.path.join(self.persist_dir, "logs", "social_feedback.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _count_hits(self, text_lower: str, vocab: set) -> int:
        hits = 0
        for w in vocab:
            pattern = re.escape(w).replace(r"\ ", r"\s+")
            if re.search(rf"(?<!\w){pattern}(?!\w)", text_lower):
                hits += 1
        return hits

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / max(1, len(letters))
