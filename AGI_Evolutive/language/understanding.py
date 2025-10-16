"""
Language v2: NLU à cadres + état de dialogue + self-ask
- Pas d'appel LLM. Heuristiques hybrides (patterns, mots-clés, scores).
- Génère des questions ciblées si incertitude élevée.
- Retourne une frame riche + propose une réponse non-générique.
"""
import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional

from AGI_Evolutive.models.intent import IntentModel
from .frames import UtteranceFrame, DialogueAct
from .dialogue_state import DialogueState


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return text


class SemanticUnderstanding:
    """
    Variante orientée dialogue avec suivi d'état et self-ask.

    Elle suppose la présence des sous-modules ``dialogue_state`` et
    ``frames`` et complète l'implémentation monolithique disponible via
    ``language.__init__`` en ajoutant la génération de questions et un
    suivi fin de l'incertitude.
    """
    def __init__(self, architecture=None, memory=None, intent_model=None, **_ignored):
        self.arch = architecture
        self.memory = memory
        self.state = DialogueState()
        if intent_model is None and architecture is not None:
            intent_model = getattr(architecture, "intent_model", None)
        self.intent_model = intent_model
        # Mini-lexique interne pour "mémoriser" des termes
        self.lexicon: Dict[str, Dict[str, Any]] = {}
        # Seuils
        self.min_intent_conf = 0.45
        self.self_ask_uncertainty = 0.35  # plus c'est grand, plus l'IA posera des questions

    # ---------- PUBLIC API ----------
    def parse_utterance(self, text: str, context: Optional[Dict[str, Any]]=None) -> UtteranceFrame:
        context = context or {}
        raw = text or ""
        norm = _normalize(raw)

        intent, conf = self._classify_intent(raw, norm)
        acts = self._guess_dialogue_acts(intent, norm)
        slots, unknowns = self._extract_slots(intent, norm)

        # Signaux d'incertitude
        uncertainty = self._compute_uncertainty(conf, unknowns, norm)

        # Enregistrer termes inconnus (pour apprentissage lexical)
        for t in unknowns:
            self._memorize_term(t)

        frame = UtteranceFrame(
            text=raw,
            normalized_text=norm,
            intent=intent,
            confidence=round(conf, 3),
            uncertainty=round(uncertainty, 3),
            acts=acts,
            slots=slots,
            unknown_terms=list(unknowns),
            needs=[]
        )

        # Mettre à jour l'état de dialogue
        self.state.update_with_frame(frame.to_dict())
        for t in unknowns:
            self.state.remember_unknown_term(t)

        # Besoins d'info (explicites)
        frame.needs = self._derive_needs(frame)

        return frame

    def respond(self, text: str, context: Optional[Dict[str, Any]]=None) -> str:
        """Produit une réponse non-générique, introspective, basée sur la frame + self-ask si besoin."""
        frame = self.parse_utterance(text, context)

        # 1) Résumé de compréhension
        summary = self._summarize_understanding(frame)

        # 2) Questions auto-générées (self-ask) si incertitude
        questions = self._self_ask(frame)
        if not questions:
            # Si on a encore des questions en attente dans l'état, on en ressort 1-2
            questions = self.state.consume_pending_questions(2)

        # 3) Micro-plan d'action (optionnel)
        plan = self._suggest_next_action(frame)

        parts = [summary]
        if questions:
            parts.append("Pour avancer, j'ai besoin de précisions : " + " | ".join(f"• {q}" for q in questions))
        if plan:
            parts.append(f"Prochain pas (proposé) : {plan}")

        return "\n".join(parts)

    # ---------- INTENT & ACTS ----------
    def _classify_intent(self, text: str, norm: Optional[str] = None) -> Tuple[str, float]:
        norm_text = norm if norm is not None else _normalize(text or "")

        if self.intent_model and hasattr(self.intent_model, "predict"):
            try:
                intent, conf = self.intent_model.predict(text)
                return intent, float(conf)
            except Exception:
                pass

        return IntentModel.rule_predict(norm_text)

    def _guess_dialogue_acts(self, intent: str, norm: str) -> List[DialogueAct]:
        acts = []
        if intent == "greet": acts.append(DialogueAct.GREET)
        if intent == "thanks": acts.append(DialogueAct.THANKS)
        if intent == "bye": acts.append(DialogueAct.BYE)
        if intent in {"request", "create", "send", "plan", "summarize", "classify"}:
            acts.append(DialogueAct.REQUEST)
        if intent in {"ask", "ask_info"}: acts.append(DialogueAct.ASK)
        if intent == "meta_help": acts.append(DialogueAct.META_HELP)
        if intent == "set_goal": acts.append(DialogueAct.INFORM)
        if intent == "reflect": acts.append(DialogueAct.REFLECT)
        if not acts:
            # heuristique : phrase déclarative
            acts.append(DialogueAct.INFORM)
        return acts

    # ---------- SLOTS & UNKNOWN TERMS ----------
    def _extract_slots(self, intent: str, norm: str) -> Tuple[Dict[str, Any], List[str]]:
        slots: Dict[str, Any] = {}
        unknowns: List[str] = []

        # termes entre guillemets → slot "quoted"
        quoted = re.findall(r"[\"""''](.*?)[\"""'']", norm)
        if quoted:
            slots["quoted"] = quoted[-1]

        # numéros / quantités
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", norm)
        if numbers:
            slots["numbers"] = numbers

        # terme après "définis/defini/definition de <concept>"
        mdef = re.search(r"(?:d[eé]finis?|d[eé]finition\s+de)\s+([a-z0-9\-_ ]{2,50})", norm)
        if mdef:
            slots["term_to_define"] = mdef.group(1).strip()

        # objectifs : rudimentaire
        if intent in ("set_goal", "request") and "goal" not in slots:
            mg = re.search(r"(?:objectif|goal|but)\s*:\s*([a-z0-9\-_ ,]{3,120})", norm)
            if mg:
                slots["goal"] = mg.group(1).strip()

        # inconnus lexicaux simples : tokens + non présents dans lexicon
        tokens = re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç0-9\-\_]{3,}", norm)
        for t in tokens:
            t_l = t.lower()
            if t_l not in self.lexicon and t_l not in {"bonjour","salut","merci","ok","fais","peux","pourquoi","comment","quoi","quand","où","but","objectif","goal","plan","aide","help","bug","erreur","super","parfait","nul","ciao","bye"}:
                # Heuristique : potentiellement intéressant / nouveau
                if len(t_l) <= 24:
                    unknowns.append(t_l)

        # éviter l'explosion : top 3
        unknowns = unknowns[:3]

        return slots, unknowns

    # ---------- INCERTITUDE, BESOINS, SELF-ASK ----------
    def _compute_uncertainty(self, intent_conf: float, unknowns: List[str], norm: str) -> float:
        u = max(0.0, 1.0 - intent_conf)
        if unknowns:
            u += 0.2 + 0.1 * (len(unknowns)-1)
        if len(norm) < 4:
            u += 0.2
        return min(1.0, u)

    def _derive_needs(self, frame: UtteranceFrame) -> List[str]:
        needs = []
        if frame.uncertainty > 0.6:
            needs.append("clarifier_objectif")
        if frame.unknown_terms:
            needs.append("definitions_termes")
        if frame.intent in ("request", "set_goal") and "goal" not in frame.slots:
            needs.append("preciser_goal")
        return needs

    def _self_ask(self, frame: UtteranceFrame) -> List[str]:
        qs: List[str] = []
        if frame.uncertainty >= self.self_ask_uncertainty:
            # Priorité aux inconnus
            for t in frame.unknown_terms:
                qs.append(f'Que signifie "{t}" dans ton contexte ?')
            # Clarif objectifs
            if "clarifier_objectif" in frame.needs:
                qs.append("Quel est le résultat concret que tu souhaites obtenir ?")
            if frame.intent == "request" and "quoted" in frame.slots:
                qs.append(f"Tu veux que j'agisse sur \"{frame.slots['quoted']}\" précisément ?")
        # Mémoriser 1-2 questions en attente
        for q in qs[:2]:
            self.state.add_pending_question(q)
        return qs[:2]

    # ---------- MÉMOIRE LEXICALE LOCALE ----------
    def _memorize_term(self, term: str):
        if term not in self.lexicon:
            self.lexicon[term] = {
                "definition": None,
                "examples": [],
                "confidence": 0.0,
                "first_seen": self.state.turn_index,
                "last_seen": self.state.turn_index,
            }
        else:
            self.lexicon[term]["last_seen"] = self.state.turn_index

    # ---------- RÉSUMÉS & PLAN D'ACTION ----------
    def _summarize_understanding(self, frame: UtteranceFrame) -> str:
        parts = []
        parts.append(f"Je pense que ton intention est **{frame.intent}** (confiance {frame.confidence:.2f}).")
        if frame.slots:
            expl = ", ".join(f"{k}={v}" for k, v in frame.slots.items())
            parts.append(f"J'ai identifié: {expl}.")
        if frame.unknown_terms:
            parts.append("J'ai des inconnus: " + ", ".join(f'"{t}"' for t in frame.unknown_terms) + ".")
        if frame.needs:
            parts.append("Besoins détectés: " + ", ".join(frame.needs) + ".")
        parts.append(f"Incertitude globale {frame.uncertainty:.2f} - je préfère vérifier avant d'agir.")
        return " ".join(parts)

    def _suggest_next_action(self, frame: UtteranceFrame) -> Optional[str]:
        # Micro plan cohérent avec l'intent
        if frame.intent == "greet":
            return "te saluer et te demander ton objectif prioritaire actuel."
        if frame.intent == "set_goal":
            return "créer/mettre à jour un objectif dans mon système si tu confirmes la formulation."
        if frame.intent == "request":
            return "décomposer la demande en étapes et vérifier que j'ai toutes les contraintes nécessaires."
        if frame.intent in {"ask", "ask_info"}:
            return "proposer une hypothèse courte puis demander validation."
        return None
