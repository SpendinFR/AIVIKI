from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable
from collections import Counter, defaultdict
import math
import os
import json
import re
import unicodedata

from AGI_Evolutive.utils.jsonsafe import json_sanitize


@dataclass
class UserStyleProfile:
    user_id: str
    avg_sentence_len: float = 14.0
    emoji_rate: float = 0.05
    exclam_rate: float = 0.03
    question_rate: float = 0.04
    formality: float = 0.6
    fav_lexicon: Dict[str, float] = None
    prefers_bullets: bool = False
    uses_caps: float = 0.05
    language: str = "fr"
    samples_seen: int = 0


class OnlineTextClassifier:
    """Very small online softmax classifier relying on bag-of-ngrams features."""

    def __init__(self, labels: Iterable[str], lr: float = 0.05):
        self.labels = tuple(labels)
        self.lr = lr
        self.weights = {label: defaultdict(float) for label in self.labels}
        self.bias = {label: 0.0 for label in self.labels}
        self.total_seen = 0

    def _tokenize(self, text: str) -> Iterable[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())

    def _features(self, text: str) -> Counter:
        tokens = list(self._tokenize(text))
        feats: Counter = Counter()
        for tok in tokens:
            feats[f"tok:{tok}"] += 1.0
            if len(tok) > 3:
                feats[f"suffix:{tok[-3:]}"] += 1.0
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            feats[f"bi:{bigram}"] += 1.0
        punctuation = Counter(
            {
                "exclam": text.count("!"),
                "question": text.count("?"),
                "ellipsis": text.count("..."),
            }
        )
        feats.update({f"punct:{k}": float(v) for k, v in punctuation.items() if v})
        emoji_count = sum(1 for ch in text if ch in StyleProfiler.EMOJI_CHARS)
        if emoji_count:
            feats["has_emoji"] = float(emoji_count)
        feats["bias"] = 1.0
        return feats

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.labels:
            return {}
        feats = self._features(text)
        scores = {}
        for label in self.labels:
            score = self.bias[label]
            weights = self.weights[label]
            for feat, value in feats.items():
                if feat in weights:
                    score += weights[feat] * value
            scores[label] = score
        max_score = max(scores.values()) if scores else 0.0
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {label: val / total for label, val in exp_scores.items()}

    def update(self, text: str, label: str):
        if label not in self.labels:
            return
        feats = self._features(text)
        probs = self.predict_proba(text)
        for lbl in self.labels:
            error = (1.0 if lbl == label else 0.0) - probs.get(lbl, 0.0)
            for feat, value in feats.items():
                self.weights[lbl][feat] += self.lr * error * value
            self.bias[lbl] += self.lr * error
        self.total_seen += 1

    def state(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "lr": self.lr,
            "weights": {lbl: dict(weights) for lbl, weights in self.weights.items()},
            "bias": dict(self.bias),
            "total_seen": self.total_seen,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]):
        classifier = cls(state.get("labels", ()), lr=state.get("lr", 0.05))
        classifier.weights = {
            lbl: defaultdict(float, weights)
            for lbl, weights in state.get("weights", {}).items()
        }
        classifier.bias = dict(state.get("bias", {})) or {lbl: 0.0 for lbl in classifier.labels}
        classifier.total_seen = state.get("total_seen", 0)
        return classifier


class StyleProfiler:
    EMOJI_CHARS = set(
        list(
            "👍👌👏💯✨🔥😄😁😊🤝❤️💪🤩🙌🌟✅🆗🙂🤗👎😡😠😞😔💢❌🛑🤬😤🙄😒😂🤣😭😅😉😎🤔🤨😍😘😉🥲🥳😇🤖"
        )
    )
    FAMILIAR_MARKERS_FR = {
        "wesh",
        "frerot",
        "frérot",
        "mdr",
        "ptdr",
        "ouais",
        "tkt",
        "dsl",
        "bg",
        "lol",
        "salut",
        "coucou",
    }
    FORMAL_MARKERS_FR = {
        "cependant",
        "toutefois",
        "ainsi",
        "par conséquent",
        "par consequent",
        "tandis que",
        "néanmoins",
        "neanmoins",
        "cordialement",
        "madame",
        "monsieur",
    }
    EN_MARKERS = {
        "the",
        "and",
        "is",
        "are",
        "you",
        "i",
        "but",
        "however",
        "therefore",
        "gonna",
        "wanna",
        "though",
        "although",
    }
    FR_LANGUAGE_REGEX = re.compile(r"\best\s+(?:un|une|le|la|l['’])", re.IGNORECASE)
    BULLET_PATTERN = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+", re.MULTILINE)

    def __init__(self, persist_path: str = "data/style_profiles.json"):
        self.persist_path = persist_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.profiles: Dict[str, UserStyleProfile] = {}
        self.global_doc_freq: Counter = Counter()
        self.total_samples = 0
        self.formality_classifier = OnlineTextClassifier(labels=("formal", "familiar"))
        self._load()

    def observe(self, user_id: str, text: str):
        p = self.profiles.get(user_id) or UserStyleProfile(user_id=user_id, fav_lexicon={})
        tokens = self._simple_tokens(text)
        token_counts = Counter(tok.lower() for tok in tokens if len(tok) > 1)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        emojis = [ch for ch in text if ch in self.EMOJI_CHARS]
        exclam = text.count("!")
        quest = text.count("?")
        caps_ratio = self._caps_ratio(text)

        lang = self._detect_language(text, tokens)

        fam = sum(1 for t in tokens if t.lower() in self.FAMILIAR_MARKERS_FR)
        formal = sum(1 for t in tokens if t.lower() in self.FORMAL_MARKERS_FR)
        formality_delta = (formal - fam) * 0.02

        classifier_adjustment = 0.0
        confident_label = None
        if abs(formal - fam) >= 2:
            confident_label = "formal" if formal > fam else "familiar"
        elif self.formality_classifier.total_seen > 10:
            proba = self.formality_classifier.predict_proba(text)
            classifier_adjustment = (proba.get("formal", 0.5) - 0.5) * 0.2

        if classifier_adjustment:
            formality_delta += classifier_adjustment

        emoji_ratio = len(emojis) / max(1, len(tokens)) if tokens else 0.0

        for tok in token_counts.keys():
            self.global_doc_freq[tok] += 1

        self.total_samples += 1

        if p.fav_lexicon is None:
            p.fav_lexicon = {}

        fav_lex = {}
        for tok, freq in token_counts.items():
            tf = freq / max(1, len(tokens))
            idf = math.log((1 + self.total_samples) / (1 + self.global_doc_freq.get(tok, 0))) + 1.0
            previous = p.fav_lexicon.get(tok, 0.0) * 0.85
            fav_lex[tok] = previous + tf * idf

        p.fav_lexicon.update(fav_lex)
        if len(p.fav_lexicon) > 60:
            top_tokens = dict(sorted(p.fav_lexicon.items(), key=lambda x: x[1], reverse=True)[:60])
            p.fav_lexicon = top_tokens

        n = p.samples_seen
        p.avg_sentence_len = (p.avg_sentence_len * n + avg_len) / (n + 1)
        p.emoji_rate = (p.emoji_rate * n + emoji_ratio) / (n + 1)
        p.exclam_rate = (p.exclam_rate * n + (exclam / max(1, len(text)))) / (n + 1)
        p.question_rate = (p.question_rate * n + (quest / max(1, len(text)))) / (n + 1)
        p.uses_caps = (p.uses_caps * n + caps_ratio) / (n + 1)
        p.formality = min(1.0, max(0.0, p.formality + formality_delta))
        p.language = lang
        p.samples_seen = n + 1

        p.prefers_bullets = p.prefers_bullets or bool(self.BULLET_PATTERN.search(text))

        self.profiles[user_id] = p

        if confident_label:
            self.formality_classifier.update(text, confident_label)

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
                data = {
                    "profiles": {uid: asdict(p) for uid, p in self.profiles.items()},
                    "_meta": {
                        "global_doc_freq": dict(self.global_doc_freq),
                        "total_samples": self.total_samples,
                        "formality_classifier": self.formality_classifier.state(),
                    },
                }
                json.dump(json_sanitize(data), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                profiles = data
                meta = {}
                if "profiles" in data:
                    profiles = data.get("profiles", {})
                    meta = data.get("_meta", {})
                for uid, d in profiles.items():
                    self.profiles[uid] = UserStyleProfile(**d)
                self.global_doc_freq = Counter(meta.get("global_doc_freq", {}))
                self.total_samples = meta.get("total_samples", len(self.profiles))
                clf_state = meta.get("formality_classifier")
                if clf_state:
                    self.formality_classifier = OnlineTextClassifier.from_state(clf_state)
        except Exception:
            self.profiles = {}
            self.global_doc_freq = Counter()
            self.total_samples = 0
            self.formality_classifier = OnlineTextClassifier(labels=("formal", "familiar"))

    def _simple_tokens(self, text: str):
        normalized = unicodedata.normalize("NFKD", text)
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", normalized)

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / max(1, len(letters))

    def _detect_language(self, text: str, tokens: Iterable[str]) -> str:
        lower_text = text.lower()
        en_hits = sum(1 for t in tokens if t.lower() in self.EN_MARKERS)
        fr_hits = sum(1 for t in tokens if t.lower() in self.FORMAL_MARKERS_FR)
        if self.FR_LANGUAGE_REGEX.search(lower_text):
            fr_hits += 2
        if en_hits > fr_hits + 1:
            return "en"
        return "fr"

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
        emojis_pos = ["🙂", "😊", "🔥", "✨", "💯", "👌", "👍", "🤝", "🙌"]
        if p.emoji_rate > 0.04 and len(txt) < 200:
            return txt + " " + emojis_pos[0]
        if p.emoji_rate > 0.08 and len(txt) < 220:
            return txt + " " + emojis_pos[min(2, len(emojis_pos) - 1)]
        return txt

    def _shape_sentence_length(self, txt: str, p: UserStyleProfile) -> str:
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        if not sents:
            return txt
        sents = [s if len(s.split()) <= 28 else self._split_long(s) for s in sents]
        return " ".join(sents)

    def _split_long(self, s: str) -> str:
        words = s.split()
        if len(words) <= 1:
            return s
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
