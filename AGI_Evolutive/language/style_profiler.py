from dataclasses import dataclass, asdict
from typing import Dict, Any
import os, json, re

from AGI_Evolutive.utils.jsonsafe import json_sanitize


@dataclass
class UserStyleProfile:
    user_id: str
    avg_sentence_len: float = 14.0
    emoji_rate: float = 0.05
    exclam_rate: float = 0.03
    question_rate: float = 0.04
    formality: float = 0.6
    fav_lexicon: Dict[str, int] = None
    prefers_bullets: bool = False
    uses_caps: float = 0.05
    language: str = "fr"
    samples_seen: int = 0


class StyleProfiler:
    EMOJI_CHARS = set(
        list("👍👌👏💯✨🔥😄😁😊🤝❤️💪🤩🙌🌟✅🆗🙂🤗👎😡😠😞😔💢❌🛑🤬😤🙄😒😂🤣😭😅😉😎🤔🤨😍😘")
    )
    FAMILIAR_MARKERS_FR = {"wesh", "frérot", "mdr", "ptdr", "ouais", "tkt", "dsl", "bg", "lol"}
    FORMAL_MARKERS_FR = {
        "cependant",
        "toutefois",
        "ainsi",
        "par conséquent",
        "tandis que",
        "néanmoins",
    }
    EN_MARKERS = {"the", "and", "is", "are", "you", "i", "but", "however", "therefore", "gonna", "wanna"}

    def __init__(self, persist_path="data/style_profiles.json"):
        self.persist_path = persist_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.profiles: Dict[str, UserStyleProfile] = {}
        self._load()

    def observe(self, user_id: str, text: str):
        p = self.profiles.get(user_id) or UserStyleProfile(user_id=user_id, fav_lexicon={})
        tokens = self._simple_tokens(text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        emojis = [ch for ch in text if ch in self.EMOJI_CHARS]
        exclam = text.count("!")
        quest = text.count("?")
        caps_ratio = self._caps_ratio(text)

        lang = "en" if sum(1 for t in tokens if t.lower() in self.EN_MARKERS) > 2 else "fr"

        fam = sum(1 for t in tokens if t.lower() in self.FAMILIAR_MARKERS_FR)
        formal = sum(1 for t in tokens if t.lower() in self.FORMAL_MARKERS_FR)
        formality_delta = (formal - fam) * 0.02

        for tok in tokens:
            if len(tok) < 2:
                continue
            p.fav_lexicon[tok.lower()] = p.fav_lexicon.get(tok.lower(), 0) + 1

        n = p.samples_seen
        p.avg_sentence_len = (p.avg_sentence_len * n + avg_len) / (n + 1)
        p.emoji_rate = (p.emoji_rate * n + (len(emojis) / max(1, len(text)))) / (n + 1)
        p.exclam_rate = (p.exclam_rate * n + (exclam / max(1, len(text)))) / (n + 1)
        p.question_rate = (p.question_rate * n + (quest / max(1, len(text)))) / (n + 1)
        p.uses_caps = (p.uses_caps * n + caps_ratio) / (n + 1)
        p.formality = min(1.0, max(0.0, p.formality + formality_delta))
        p.language = lang
        p.samples_seen = n + 1

        p.prefers_bullets = p.prefers_bullets or bool(re.search(r"^- |\d+\)", text, re.M))

        self.profiles[user_id] = p
        self._save()

    def style_of(self, user_id: str) -> UserStyleProfile:
        return self.profiles.get(user_id) or UserStyleProfile(user_id=user_id, fav_lexicon={})

    def rewrite_to_match(self, base_text: str, user_id: str) -> str:
        p = self.style_of(user_id)

        if p.language == "fr":
            base_text = self._adjust_formality_fr(base_text, p.formality)

        base_text = self._shape_punctuation(base_text, p)

        if p.emoji_rate > 0.01:
            base_text = self._sprinkle_emojis(base_text, p)

        base_text = self._shape_sentence_length(base_text, p)

        if p.uses_caps > 0.12:
            base_text = self._emphasize_some_words(base_text)

        return base_text

    def _save(self):
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                data = {uid: asdict(p) for uid, p in self.profiles.items()}
                json.dump(json_sanitize(data), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for uid, d in data.items():
                    self.profiles[uid] = UserStyleProfile(**d)
        except Exception:
            self.profiles = {}

    def _simple_tokens(self, text: str):
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / max(1, len(letters))

    def _adjust_formality_fr(self, txt: str, formality: float) -> str:
        replacements_familiar = {
            "bonjour": "salut",
            "je vais": "j'vais",
            "je ne": "j'",
            "ne pas": "pas",
        }
        replacements_formal = {
            "salut": "bonjour",
            "ok": "d'accord",
            "ouais": "oui",
            "t'inquiète": "ne vous en faites pas",
        }
        if formality < 0.45:
            for k, v in replacements_familiar.items():
                txt = re.sub(rf"\b{k}\b", v, txt, flags=re.I)
        elif formality > 0.65:
            for k, v in replacements_formal.items():
                txt = re.sub(rf"\b{k}\b", v, txt, flags=re.I)
        return txt

    def _shape_punctuation(self, txt: str, p: UserStyleProfile) -> str:
        if p.exclam_rate > 0.01 and not txt.strip().endswith("!"):
            txt = txt.rstrip(". ") + "!"
        if p.question_rate > 0.02 and not txt.strip().endswith("?"):
            if len(txt) < 140:
                txt += " (tu vois ?)" if p.language == "fr" else " (you see?)"
        return txt

    def _sprinkle_emojis(self, txt: str, p: UserStyleProfile) -> str:
        EMOJIS_POS = ["🙂", "😊", "🔥", "✨", "💯", "👌", "👍", "🤝", "🙌"]
        if p.emoji_rate > 0.04 and len(txt) < 200:
            return txt + " " + EMOJIS_POS[0]
        return txt

    def _shape_sentence_length(self, txt: str, p: UserStyleProfile) -> str:
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        if not sents:
            return txt
        target = p.avg_sentence_len
        sents = [s if len(s.split()) <= 28 else self._split_long(s) for s in sents]
        return " ".join(sents)

    def _split_long(self, s: str) -> str:
        words = s.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + ". " + " ".join(words[mid:])

    def _emphasize_some_words(self, txt: str) -> str:
        tokens = self._simple_tokens(txt)
        if len(tokens) < 6:
            return txt
        idxs = [i for i, t in enumerate(tokens) if len(t) > 4]
        if not idxs:
            return txt
        for i in idxs[:2]:
            tokens[i] = tokens[i].upper()
        return " ".join(tokens)
