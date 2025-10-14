
# language/__init__.py
"""
Module Language — auto‑contenu, optimisé et complet pour ton AGI.
Contient :
  - SemanticUnderstanding  : parsing sémantique (frames, intents, slots, entities)
  - PragmaticReasoning     : contexte, intentions, implicatures, actes de langage
  - DiscourseProcessing    : gestion de la cohérence inter‑tour, anaphores simples
  - LanguageGeneration     : génération textuelle contrôlée par but/tonalité/style

Objectifs :
  - AUCUN import de sous‑fichier (fini les ModuleNotFoundError)
  - Auto‑wiring doux via cognitive_architecture (getattr, sans import croisé)
  - Persistance (to_state / from_state) pour travailler offline et reprendre
  - Standard library only (pas de dépendances externes)

N.B. : Ce module est “optimisé mais complet” : toute la logique utile est là,
avec des implémentations compactes et robustes (pas de verbiage inutile).
"""

from __future__ import annotations
import re, time, math, random, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Utilitaires
# ============================================================

def _now() -> float:
    return time.time()

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _mean(xs: List[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


# ============================================================
# Types et structures de base
# ============================================================

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    canonical: Optional[str] = None

@dataclass
class Frame:
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.6

@dataclass
class Utterance:
    surface_form: str
    lang: str = "fr"
    tokens: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    frame: Optional[Frame] = None
    pragmatics: Dict[str, Any] = field(default_factory=dict)  # act, politeness, uncertainty
    timestamp: float = field(default_factory=_now)


# ============================================================
# 1) SemanticUnderstanding
# ============================================================

class SemanticUnderstanding:
    """
    Compréhension sémantique compacte :
     - tokenisation simple, NER par regex (dates, nombres, emails, urls, montants)
     - frame intent+slots : détecte objectifs fréquents (demander, informer, créer, planifier, envoyer)
     - calcul d'incertitude : hedges (“peut‑être”, “je crois”), tournures interrogatives
    """
    RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    RE_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
    RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
    RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
    RE_MONEY = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:€|eur|euros|\$|usd)\b", re.I)

    HEDGES = {"peut-être", "je crois", "il me semble", "probablement", "possiblement", "éventuellement"}
    INTENT_PATTERNS = [
        ("ask_info", re.compile(r"\b(quoi|quel|quelle|quels|comment|pourquoi|combien|où|quand)\b|\?$", re.I)),
        ("create", re.compile(r"\b(crée|creer|ajoute|ajouter|enregistre|note)\b", re.I)),
        ("send", re.compile(r"\b(envoie|envoyer|partage|transmets?)\b", re.I)),
        ("plan", re.compile(r"\b(planifie|planifier|prévois|organise|agenda)\b", re.I)),
        ("inform", re.compile(r"\b(j'informe|voici|sache que|pour info|FYI)\b", re.I)),
        ("summarize", re.compile(r"\b(résume|resume|synthétise|synthese|summary)\b", re.I)),
        ("classify", re.compile(r"\b(classe|catégorise|tague|étiquette)\b", re.I)),
    ]

    def __init__(self, cognitive_architecture: Any = None, memory_system: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.arch = cognitive_architecture
        self.memory_system = memory_system
        self.history: List[Utterance] = []
        self.lang = "fr"

        # auto‑wiring : accès doux aux autres modules si dispos
        ca = self.cognitive_arch
        if ca:
            self.reasoning = getattr(ca, "reasoning", None)
            self.goals = getattr(ca, "goals", None)
            self.emotions = getattr(ca, "emotions", None)
            self.metacognition = getattr(ca, "metacognition", None)
            self.world_model = getattr(ca, "world_model", None)
            self.perception = getattr(ca, "perception", None)
            self.creativity = getattr(ca, "creativity", None)

    # --------- Pipeline principal ---------

    def parse_utterance(self, text: str, context: Optional[Dict[str, Any]] = None) -> Utterance:
        context = context or {}
        toks = self._tokenize(text)
        ents = self._ner(text)
        frame = self._frame(text, toks, ents)
        prag = self._pragmatics(text, toks)

        utt = Utterance(surface_form=text, lang=self.lang, tokens=toks, entities=ents, frame=frame, pragmatics=prag)
        self.history.append(utt)
        if len(self.history) > 200:
            self.history.pop(0)
        # informer la méta du niveau d’incertitude
        if getattr(self, "metacognition", None) and hasattr(self.metacognition, "register_language_parse"):
            try:
                self.metacognition.register_language_parse(utt.frame.confidence if utt.frame else 0.3, prag.get("uncertainty", 0.0))
            except Exception:
                pass
        return utt

    def respond(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Parse l’énoncé utilisateur et orchestre un raisonnement léger pour répondre."""
        context = context or {}
        utterance = self.parse_utterance(user_text, context)
        frame = getattr(utterance, "frame", None)

        arch = getattr(self, "arch", None) or getattr(self, "cognitive_arch", None)
        reasoner = getattr(arch, "reasoning", None) if arch else None
        if reasoner is None:
            reasoner = getattr(self, "reasoning", None)

        reasoned = None
        if reasoner and hasattr(reasoner, "reason"):
            try:
                q = (
                    getattr(utterance, "surface_form", None)
                    or getattr(frame, "normalized_text", None)
                    or user_text
                )
                reasoned = reasoner.reason(q, context={"frame_intent": getattr(frame, "intent", "")})
            except Exception:
                reasoned = None

        if reasoned:
            steps_lines = [
                f"– {t['strategy']}: {t['notes']}"
                for t in reasoned.get("trace", [])
                if t.get("notes")
            ]
            steps = "\n".join(steps_lines) if steps_lines else "– (aucune note)"
            support = reasoned.get("support", [])
            support_lines = (
                "\nContexte mémoire:\n" + "\n".join([f"• {s}" for s in support])
            ) if support else ""
            meta = reasoned.get("meta") or {}
            confidence = float(reasoned.get("confidence", 0.0))
            complexity = float(meta.get("complexity", 0.0))
            response = (
                f"{reasoned.get('answer', '')}\n"
                f"(confiance ~ {confidence:.2f}, complexité {complexity:.2f})\n"
                f"Démarche:\n{steps}{support_lines}"
            )
        else:
            response = f"Reçu: {getattr(utterance, 'surface_form', user_text)}"

        return response

    # --------- Étapes internes ---------

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE) if t.strip()]

    def _ner(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        for r, label in ((self.RE_DATE, "DATE"), (self.RE_EMAIL, "EMAIL"), (self.RE_URL, "URL"),
                         (self.RE_MONEY, "MONEY"), (self.RE_NUMBER, "NUMBER")):
            for m in r.finditer(text):
                ents.append(Entity(text=m.group(0), label=label, start=m.start(), end=m.end()))
        return ents

    def _frame(self, text: str, toks: List[str], ents: List[Entity]) -> Frame:
        intent = "inform"
        conf = 0.55
        for it, pat in self.INTENT_PATTERNS:
            if pat.search(text):
                intent, conf = it, 0.75
                break
        slots: Dict[str, Any] = {}
        # Remonter quelques entités utiles en slots
        for e in ents:
            if e.label in ("DATE", "MONEY", "URL", "EMAIL", "NUMBER"):
                slots.setdefault(e.label.lower()+"s", []).append(e.text)
        # Heuristique de cible/objet
        m = re.search(r"\b(?:sur|à propos de|concernant)\s+(.+)$", text, re.I)
        if m:
            slots["topic"] = m.group(1).strip()[:120]
        return Frame(intent=intent, slots=slots, confidence=conf)

    def _pragmatics(self, text: str, toks: List[str]) -> Dict[str, Any]:
        t = text.strip().lower()
        act = "statement"
        if t.endswith("?"):
            act = "question"
        elif t.endswith("!"):
            act = "exclaim"
        # Politeness & uncertainty
        polite = any(w in t for w in ("s'il te plaît", "svp", "merci"))
        uncertainty = 0.2 if any(h in t for h in self.HEDGES) else 0.0
        return {"speech_act": act, "politeness": polite, "uncertainty": uncertainty}

    # --------- Persistance ---------

    def to_state(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "history": [{
                "surface": u.surface_form,
                "lang": u.lang,
                "tokens": u.tokens,
                "entities": [e.__dict__ for e in u.entities],
                "frame": u.frame.__dict__ if u.frame else None,
                "pragmatics": u.pragmatics,
                "timestamp": u.timestamp,
            } for u in self.history[-100:]]
        }

    def from_state(self, state: Dict[str, Any]):
        self.lang = state.get("lang", "fr")
        self.history = []
        for d in state.get("history", []):
            ents = [Entity(**e) for e in d.get("entities", [])]
            fr = d.get("frame")
            fr = Frame(**fr) if fr else None
            self.history.append(Utterance(
                surface_form=d.get("surface", ""),
                lang=d.get("lang", "fr"),
                tokens=d.get("tokens", []),
                entities=ents,
                frame=fr,
                pragmatics=d.get("pragmatics", {}),
                timestamp=d.get("timestamp", _now()),
            ))


# ============================================================
# 2) PragmaticReasoning
# ============================================================

class PragmaticReasoning:
    """
    Raisonner au‑delà du littéral :
     - infère l’intention et les présupposés (ex : “tu peux… ?” = requête)
     - ajuste au contexte (état émotionnel, objectifs actifs, normes sociales)
     - propose des actes de langage appropriés (répondre, demander précision, proposer plan)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.context: Dict[str, Any] = {
            "tone": "neutral",
            "formality": 0.5,
            "cooperation": 0.7,
            "confidence": 0.6,
        }
        # auto‑wiring
        ca = self.cognitive_arch
        if ca:
            self.emotions = getattr(ca, "emotions", None)
            self.goals = getattr(ca, "goals", None)
            self.world_model = getattr(ca, "world_model", None)
            self.metacognition = getattr(ca, "metacognition", None)

    def infer_intent(self, utt: Utterance) -> Dict[str, Any]:
        # base sur frame + ajustements pragm.
        intent = utt.frame.intent if utt.frame else "inform"
        act = utt.pragmatics.get("speech_act", "statement")
        uncertainty = utt.pragmatics.get("uncertainty", 0.0)
        # Implicature simple : question indirecte
        if re.search(r"\b(peux-tu|pourrais-tu|tu peux|tu pourrais)\b", utt.surface_form.lower()):
            intent, act = "request", "question"
        # Ajustement par émotions globales si dispo
        if getattr(self, "emotions", None) and hasattr(self.emotions, "current_valence"):
            try:
                val = float(self.emotions.current_valence)
                self.context["tone"] = "upbeat" if val > 0.3 else ("down" if val < -0.3 else "neutral")
            except Exception:
                pass
        return {"intent": intent, "speech_act": act, "uncertainty": uncertainty, "context": dict(self.context)}

    def next_action(self, pragmatic: Dict[str, Any]) -> str:
        intent, act, uncert = pragmatic["intent"], pragmatic["speech_act"], pragmatic["uncertainty"]
        if act == "question":
            return "answer_or_ask_clarification"
        if intent in ("create", "plan", "send"):
            return "propose_plan"
        if uncert > 0.3:
            return "ask_clarification"
        return "answer"

    def to_state(self) -> Dict[str, Any]:
        return {"context": dict(self.context)}

    def from_state(self, state: Dict[str, Any]):
        self.context.update(state.get("context", {}))


# ============================================================
# 3) DiscourseProcessing
# ============================================================

@dataclass
class DiscourseState:
    last_entities: List[Entity] = field(default_factory=list)
    last_topics: List[str] = field(default_factory=list)
    turn_index: int = 0

class DiscourseProcessing:
    """
    Gestion de discours minimale :
     - suit les entités/objets récurrents (anaphores très simples)
     - maintient un “topic stack” de 5 éléments max
     - fournit contexte de réponse (“réutiliser la dernière URL”, etc.)
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.state = DiscourseState()

    def update(self, utt: Utterance):
        self.state.turn_index += 1
        ents = [e for e in utt.entities if e.label in ("URL", "EMAIL", "MONEY", "DATE", "NUMBER")]
        topics = []
        if utt.frame and "topic" in utt.frame.slots:
            topics.append(utt.frame.slots["topic"])
        # push
        self.state.last_entities = (self.state.last_entities + ents)[-10:]
        self.state.last_topics = (self.state.last_topics + topics)[-5:]

    def resolve_reference(self, text: str) -> Dict[str, Any]:
        # “celui‑ci”, “ça”, “ce document”, “la même url”, etc.
        t = text.lower()
        ref_ent = None
        if "même url" in t or "la même url" in t:
            ref_ent = next((e for e in reversed(self.state.last_entities) if e.label == "URL"), None)
        return {"resolved_entity": ref_ent.__dict__ if ref_ent else None, "topic": (self.state.last_topics[-1] if self.state.last_topics else None)}

    def to_state(self) -> Dict[str, Any]:
        return {
            "turn_index": self.state.turn_index,
            "last_entities": [e.__dict__ for e in self.state.last_entities],
            "last_topics": list(self.state.last_topics),
        }

    def from_state(self, state: Dict[str, Any]):
        self.state.turn_index = int(state.get("turn_index", 0))
        self.state.last_entities = [Entity(**e) for e in state.get("last_entities", [])]
        self.state.last_topics = list(state.get("last_topics", []))


# ============================================================
# 4) LanguageGeneration
# ============================================================

class LanguageGeneration:
    """
    Génération textuelle pragmatique :
     - templates intelligents + mix lexical (tonalité, style, concision)
     - prise en compte de l’incertitude et du besoin de clarification
     - formats utilitaires : listes, plans d’action, réponses directes
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.style = {"formality": 0.5, "warmth": 0.6, "conciseness": 0.7}
        # auto‑wiring
        ca = self.cognitive_arch
        if ca:
            self.language = getattr(ca, "language", None)  # self‑ref possible
            self.goals = getattr(ca, "goals", None)
            self.metacognition = getattr(ca, "metacognition", None)

    def reply(self, intent: str, data: Dict[str, Any], pragmatic: Dict[str, Any]) -> str:
        act = pragmatic.get("speech_act", "statement")
        ctx = pragmatic.get("context", {})
        tone = ctx.get("tone", "neutral")
        uncertainty = pragmatic.get("uncertainty", 0.0)
        topic = data.get("topic")
        # Stratégies simples mais efficaces
        if intent == "ask_info" or act == "question":
            return self._answer_question(data, tone=tone)
        if intent in ("create", "plan", "send"):
            return self._propose_plan(intent, data, tone=tone)
        if uncertainty > 0.3:
            return self._ask_clarification(topic, tone=tone)
        # par défaut : réponse informative
        return self._inform(data, tone=tone)

    # ----- patterns -----

    def _answer_question(self, data: Dict[str, Any], tone: str = "neutral") -> str:
        topic = data.get("topic")
        if topic:
            base = f"Pour {topic}, voici ce que je vois."
        else:
            base = "Voici ce que je peux te dire."
        hints = data.get("hints", [])
        if hints:
            base += " " + " ".join(hints[:3])
        return self._tone(base, tone)

    def _propose_plan(self, intent: str, data: Dict[str, Any], tone: str = "neutral") -> str:
        topic = data.get("topic", "la tâche")
        steps = data.get("steps") or ["Clarifier l’objectif", "Rassembler les données utiles", "Proposer une ébauche", "Itérer avec ton feedback"]
        head = f"Je te propose un mini‑plan pour {topic} :"
        bullet = "\n- " + "\n- ".join(steps[:6])
        return self._tone(head + bullet, tone)

    def _ask_clarification(self, topic: Optional[str], tone: str = "neutral") -> str:
        if topic:
            msg = f"Peux‑tu préciser les contraintes principales pour {topic} ?"
        else:
            msg = "Peux‑tu préciser le contexte ou les contraintes principales ?"
        return self._tone(msg, tone)

    def _inform(self, data: Dict[str, Any], tone: str = "neutral") -> str:
        parts = []
        if "topic" in data:
            parts.append(f"À propos de {data['topic']},")
        if "summary" in data:
            parts.append(str(data["summary"]))
        elif "text" in data:
            parts.append(str(data["text"])[:300])
        else:
            parts.append("c’est noté.")
        return self._tone(" ".join(parts), tone)

    # ----- utils -----

    def _tone(self, text: str, tone: str) -> str:
        if tone == "upbeat":
            return "🙂 " + text
        if tone == "down":
            return "… " + text
        return text

    def to_state(self) -> Dict[str, Any]:
        return {"style": dict(self.style)}

    def from_state(self, state: Dict[str, Any]):
        self.style.update(state.get("style", {}))


# ============================================================
# Export public
# ============================================================

__all__ = [
    "SemanticUnderstanding",
    "LanguageGeneration",
    "PragmaticReasoning",
    "DiscourseProcessing",
    "Utterance",
    "Frame",
    "Entity",
]
