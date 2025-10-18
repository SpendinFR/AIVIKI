# language/__init__.py
"""
Module Language - auto-contenu, optimisé et complet pour ton AGI.
Contient :
  - SemanticUnderstanding  : parsing sémantique (frames, intents, slots, entities)
  - PragmaticReasoning     : contexte, intentions, implicatures, actes de langage
  - DiscourseProcessing    : gestion de la cohérence inter-tour, anaphores simples
  - LanguageGeneration     : génération textuelle contrôlée par but/tonalité/style

Objectifs :
  - AUCUN import de sous-fichier (fini les ModuleNotFoundError)
  - Auto-wiring doux via cognitive_architecture (getattr, sans import croisé)
  - Persistance (to_state / from_state) pour travailler offline et reprendre
  - Standard library only (pas de dépendances externes)

N.B. : Ce module est "optimisé mais complet" : toute la logique utile est là,
avec des implémentations compactes et robustes (pas de verbiage inutile).
"""

from __future__ import annotations
import os, re, time, math, random, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from AGI_Evolutive.models.intent import IntentModel


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)


def _json_load(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _json_save(path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
    normalized_text: str = ""


# ============================================================
# 1) SemanticUnderstanding
# ============================================================

class SemanticUnderstanding:
    """
    Compréhension sémantique compacte :
     - tokenisation simple, NER par regex (dates, nombres, emails, urls, montants)
     - frame intent+slots : détecte objectifs fréquents (demander, informer, créer, planifier, envoyer)
     - calcul d'incertitude : hedges ("peut-être", "je crois"), tournures interrogatives

    Cette implémentation monolithique permet d'importer ``AGI_Evolutive.language``
    sans dépendances supplémentaires.  Le fichier
    :mod:`AGI_Evolutive.language.understanding` propose, lui, une variante
    plus modulaire centrée sur l'état de dialogue et les questions de
    clarification.
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

    def __init__(
        self,
        cognitive_architecture: Any = None,
        memory_system: Any = None,
        intent_model: Optional[IntentModel] = None,
    ):
        self.cognitive_arch = cognitive_architecture
        self.arch = cognitive_architecture
        self.memory_system = memory_system
        self.memory = memory_system  # alias pour compatibilité/reflexive replies
        self.history: List[Utterance] = []
        self.lang = "fr"

        if intent_model is None and cognitive_architecture is not None:
            intent_model = getattr(cognitive_architecture, "intent_model", None)
        self.intent_model: IntentModel = intent_model or IntentModel()

        # auto-wiring : accès doux aux autres modules si dispos
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
        normalized = " ".join(toks).lower()

        utt = Utterance(
            surface_form=text,
            lang=self.lang,
            tokens=toks,
            entities=ents,
            frame=frame,
            pragmatics=prag,
            normalized_text=normalized,
        )
        self.history.append(utt)
        if len(self.history) > 200:
            self.history.pop(0)
        # informer la méta du niveau d'incertitude
        if getattr(self, "metacognition", None) and hasattr(
            self.metacognition, "register_language_parse"
        ):
            try:
                self.metacognition.register_language_parse(
                    utt.frame.confidence if utt.frame else 0.3,
                    prag.get("uncertainty", 0.0),
                )
            except Exception:
                pass
        return utt

    def respond(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Produit une réponse textuelle tout en fournissant un résumé de l'analyse."""
        context = context or {}
        parsed = self.parse_utterance(user_text, context)
        frame = getattr(parsed, "frame", None)

        arch = getattr(self, "arch", None) or getattr(self, "cognitive_arch", None)
        reasoner = getattr(arch, "reasoning", None) if arch else None
        if reasoner is None:
            reasoner = getattr(self, "reasoning", None)

        reasoned = None
        if reasoner and hasattr(reasoner, "reason"):
            try:
                question = (
                    getattr(parsed, "surface_form", None)
                    or getattr(frame, "normalized_text", None)
                    or user_text
                )
                reasoned = reasoner.reason(
                    question,
                    context={"frame_intent": getattr(frame, "intent", "")},
                )
            except Exception:
                reasoned = None

        contexts = self._retrieve_context(parsed)

        if frame is None and isinstance(parsed, Frame):
            frame = parsed

        intent = getattr(frame, "intent", "inform")
        confidence = float(getattr(frame, "confidence", 0.5) or 0.0)
        slots = getattr(frame, "slots", {}) or {}

        slot_bits: List[str] = []
        for key, val in slots.items():
            if isinstance(val, (list, tuple)):
                slot_bits.append(f"{key}=" + ", ".join(str(v) for v in val))
            else:
                slot_bits.append(f"{key}={val}")

        summary = f"Intent détecté: {intent} (confiance {confidence:.2f})."
        if slot_bits:
            summary += " Slots: " + "; ".join(slot_bits)

        context_lines: List[str] = []
        if contexts:
            for c in contexts:
                label = c.get("title") or "mémoire"
                context_lines.append(f"• ({c['score']:.2f}) {label}: {c['snippet']}")

        contextualised_summary_parts = [summary]
        if context_lines:
            contextualised_summary_parts.append(
                "Contexte pertinent retrouvé:\n" + "\n".join(context_lines)
            )
        contextualised_summary = "\n\n".join(contextualised_summary_parts)

        if reasoned:
            steps_lines = [
                f"- {t['strategy']}: {t['notes']}"
                for t in reasoned.get("trace", [])
                if t.get("notes")
            ]
            steps = "\n".join(steps_lines) if steps_lines else "- (aucune note)"
            support = reasoned.get("support", [])
            support_lines = (
                "\nContexte mémoire:\n" + "\n".join([f"• {s}" for s in support])
            ) if support else ""
            meta = reasoned.get("meta") or {}
            confidence = float(reasoned.get("confidence", 0.0))
            complexity = float(meta.get("complexity", 0.0))
            answer = (
                f"{reasoned.get('answer', '')}\n"
                f"(confiance ~ {confidence:.2f}, complexité {complexity:.2f})\n"
                f"Démarche:\n{steps}{support_lines}"
            )
            if contextualised_summary:
                return f"{answer}\n\n{contextualised_summary}"
            return answer

        fallback = getattr(parsed, "surface_form", user_text)
        base_response = f"Reçu: {fallback}"
        if contextualised_summary:
            return f"{base_response}\n\n{contextualised_summary}"
        return base_response

    def _retrieve_context(self, frame) -> List[dict]:
        """
        Cherche 3 éléments pertinents (interactions + docs) à partir du texte normalisé
        et des slots importants. Tolérant si memory absent.
        """
        results: List[dict] = []
        try:
            arch = getattr(self, "arch", None)
            mem = getattr(self, "memory_system", None) or (getattr(arch, "memory", None) if arch else None)
            retr = getattr(mem, "retrieval", None)
            if not retr:
                return results

            q = getattr(frame, "normalized_text", "") or getattr(frame, "surface_form", "")
            extras: List[str] = []

            slots_obj = getattr(frame, "slots", None)
            if slots_obj is None and getattr(frame, "frame", None):
                slots_obj = getattr(frame.frame, "slots", None)

            if isinstance(slots_obj, dict):
                for k in ("quoted", "goal", "term_to_define"):
                    if k in slots_obj:
                        val = slots_obj[k]
                        if isinstance(val, (list, tuple)):
                            extras.extend([str(v) for v in val])
                        else:
                            extras.append(str(val))
            if extras:
                q = (q + " " + " ".join(extras)).strip()

            if not q:
                return results

            hits = retr.search_text(q, top_k=3)
            for h in hits:
                txt = h.get("text", "")
                meta = h.get("meta", {})
                snippet = (txt[:180] + "…") if len(txt) > 180 else txt
                title = meta.get("title") or meta.get("source") or meta.get("type", "")
                results.append({
                    "score": h.get("score", 0.0),
                    "title": title,
                    "snippet": snippet,
                })
        except Exception:
            return []
        return results

    def generate_reflective_reply(self, arch, user_msg: str) -> str:
        """Génère une réponse réflexive courte et lisible."""
        status = {}
        try:
            status = arch.get_cognitive_status()
        except Exception:
            status = {}

        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", user_msg.lower())
        unknown = []
        try:
            memory = getattr(self, "memory", None) or getattr(self, "memory_system", None)
            if memory and hasattr(memory, "knows"):
                unknown = [t for t in tokens if not memory.knows(t)]
            elif memory and hasattr(memory, "retrieve"):
                for t in tokens:
                    try:
                        res = memory.retrieve(t, top_k=1)
                        if not res:
                            unknown.append(t)
                    except Exception:
                        pass
            else:
                unknown = [t for t in tokens if len(t) > 12]
        except Exception:
            pass
        unknown = list(dict.fromkeys(unknown))[:3]

        reasoning = status.get("reasoning", {})
        creativity = status.get("creativity", {})
        metacog = status.get("metacognition", {})
        activation = status.get("global_activation", 0.5)

        doing = []
        if reasoning.get("recent_inferences", 0) > 0:
            doing.append("j'analyse des inférences récentes")
        if creativity.get("recent_ideas", 0) > 0:
            doing.append("je génère/évalue de nouvelles idées")
        if metacog.get("events", 0) > 0:
            doing.append("je surveille mes performances et erreurs")
        if not doing:
            doing.append("je collecte des repères pour mieux te comprendre")

        confusion = []
        if unknown:
            confusion.append("je ne suis pas sûr de bien cerner: " + ", ".join(unknown))
        avg_conf = reasoning.get("avg_confidence", 0.5)
        if isinstance(avg_conf, (int, float)) and avg_conf < 0.45:
            confusion.append("ma confiance de raisonnement est un peu basse sur ce sujet")

        next_steps = []
        if unknown:
            next_steps.append("tu peux m'expliquer ce que tu entends par " + ", ".join(unknown) + " ?")
        else:
            next_steps.append("je peux tenter un exemple concret ou reformuler si tu veux")
        if isinstance(activation, (int, float)) and activation < 0.4:
            next_steps.append("je vais réduire la complexité pour m'aligner")

        lines = []
        lines.append("• Ce que je fais: " + "; ".join(doing))
        if confusion:
            lines.append("• Ce que je ne comprends pas: " + "; ".join(confusion))
        lines.append("• Ce que je propose: " + "; ".join(next_steps))

        try:
            gist = user_msg.strip()
            if len(gist) > 120:
                gist = gist[:117] + "…"
            lines.append(f"• Ta demande (j'ai bien reçu): \"{gist}\"")
        except Exception:
            pass

        return "\n".join(lines)

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
        if self.intent_model and hasattr(self.intent_model, "predict"):
            try:
                intent, conf = self.intent_model.predict(text)
            except Exception:
                intent, conf = IntentModel.rule_predict(text)
        else:
            intent, conf = IntentModel.rule_predict(text)

        if intent == "inform":
            for it, pat in self.INTENT_PATTERNS:
                if pat.search(text):
                    intent, conf = it, max(conf, 0.75)
                    break

        try:
            conf = float(conf)
        except Exception:
            conf = 0.55
        conf = max(0.0, min(1.0, conf))
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
                "normalized_text": u.normalized_text,
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
                normalized_text=d.get("normalized_text", ""),
            ))


# ============================================================
# 2) PragmaticReasoning
# ============================================================

class PragmaticReasoning:
    """
    Raisonner au-delà du littéral :
     - infère l'intention et les présupposés (ex : "tu peux… ?" = requête)
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
        # auto-wiring
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
     - maintient un "topic stack" de 5 éléments max
     - fournit contexte de réponse ("réutiliser la dernière URL", etc.)
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
        # "celui-ci", "ça", "ce document", "la même url", etc.
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
     - prise en compte de l'incertitude et du besoin de clarification
     - formats utilitaires : listes, plans d'action, réponses directes
    """
    def __init__(self, cognitive_architecture: Any = None):
        self.cognitive_arch = cognitive_architecture
        self.style = {"formality": 0.5, "warmth": 0.6, "conciseness": 0.7}
        # auto-wiring
        ca = self.cognitive_arch
        if ca:
            self.language = getattr(ca, "language", None)  # self-ref possible
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
        steps = data.get("steps") or ["Clarifier l'objectif", "Rassembler les données utiles", "Proposer une ébauche", "Itérer avec ton feedback"]
        head = f"Je te propose un mini-plan pour {topic} :"
        bullet = "\n- " + "\n- ".join(steps[:6])
        return self._tone(head + bullet, tone)

    def _ask_clarification(self, topic: Optional[str], tone: str = "neutral") -> str:
        if topic:
            msg = f"Peux-tu préciser les contraintes principales pour {topic} ?"
        else:
            msg = "Peux-tu préciser le contexte ou les contraintes principales ?"
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
            parts.append("c'est noté.")
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
