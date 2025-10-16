from __future__ import annotations

from typing import Dict, Any, List, Tuple
import random
import re
import time

from AGI_Evolutive.social.tactic_selector import TacticSelector
from AGI_Evolutive.social.interaction_rule import ContextBuilder


def _tokens(s: str) -> set:
    return set(re.findall(r"[A-Za-zÀ-ÿ]{3,}", (s or "").lower()))


class LanguageRenderer:
    def __init__(self, voice_profile, lexicon):
        self.voice = voice_profile
        self.lex = lexicon
        # anti-spam / fréquence
        self._cooldown = {"past": 0.0, "colloc": 0.0}
        self._last_used = {"past": "", "colloc": ""}

        # seuils réglables
        self.THRESH = {
            "past_relevance": 0.25,   # pertinence mini lien passé
            "colloc_relevance": 0.20, # pertinence mini collocation
            "conf_min": 0.55,         # confiance mini pour ornement
            "chance_colloc": 0.35,    # proba de tenter une collocation (modulée)
        }

    # ---------- utilitaires ----------
    def _confidence(self) -> float:
        # essaie d’extraire une confiance globale depuis la policy
        try:
            pol = self.voice.self_model.arch.policy
            if hasattr(pol, "confidence"):
                return float(pol.confidence())
        except Exception:
            pass
        return 0.6

    def _budget_chars(self, ctx: Dict[str, Any]) -> int:
        # budget d’ornement selon style utilisateur / voix
        st = self.voice.style()
        user = (ctx.get("user_style") or {})
        base = 160  # budget max d’ornement
        if user.get("prefers_long"):
            base += 80
        if st.get("conciseness", 0.6) > 0.7:
            base -= 60
        return max(0, base)

    def _decrease_cooldowns(self) -> None:
        # refroidit un peu à chaque appel
        for k in self._cooldown:
            self._cooldown[k] = max(0.0, self._cooldown[k] - 0.34)

    # ---------- sélection “lien au passé” ----------
    def _pick_relevant_moment(self, user_msg: str, ctx: Dict[str, Any]) -> Tuple[str, float]:
        moments: List[str] = ctx.get("key_moments") or []
        if not moments:
            return ("", 0.0)
        utoks = _tokens(user_msg)
        best, score = "", 0.0
        for m in moments[-8:]:
            mtoks = _tokens(m)
            if not mtoks:
                continue
            jacc = len(utoks & mtoks) / max(1, len(utoks | mtoks))
            if jacc > score:
                best, score = m, jacc
        return (best, score)

    # ---------- sélection “collocation aimée” ----------
    def _pick_collocation(self, ctx: Dict[str, Any]) -> Tuple[str, float]:
        topics = set(ctx.get("topics") or [])
        cand = self.lex.sample_collocation(novelty=0.3)
        if not cand:
            return ("", 0.0)
        rel = len(_tokens(cand) & topics) / max(1, len(_tokens(cand)))
        return (cand, rel)

    # ---------- décor et rendu ----------
    def _decorate_with_voice(self, text: str) -> str:
        st = self.voice.style()
        if st.get("formality", 0.4) > 0.75 and not text.lower().startswith(("bonjour", "bonsoir")):
            text = "Bonjour, " + text
        if st.get("warmth", 0.6) > 0.75 and not text.endswith("!"):
            text += " (je reste dispo si besoin)"
        if st.get("emoji", 0.2) > 0.6:
            text = "🙂 " + text
        return text

    def render_reply(self, semantics: Dict[str, Any], ctx: Dict[str, Any]) -> str:
        """
        Règle : on n’ajoute un lien au passé / une collocation QUE si :
        - confiance >= conf_min
        - pertinence >= seuils
        - budget > 0
        - cooldown OK et pas de doublon
        Et au plus 1 ornement par réponse (priorité au lien passé).
        """
        self._decrease_cooldowns()
        base = (semantics.get("text") or "").strip() or "Je te réponds en tenant compte de notre historique."
        conf = self._confidence()
        budget = self._budget_chars(ctx)

        # Si la question est directe / l’utilisateur veut court → pas d’ornement
        is_direct_question = "?" in (ctx.get("last_message") or "")
        if conf < self.THRESH["conf_min"] or budget < 60 or (ctx.get("user_style") or {}).get("prefers_long") is False:
            return self._decorate_with_voice(base)

        # 1) Candidat lien au passé
        past_txt, past_rel = self._pick_relevant_moment(ctx.get("last_message", ""), ctx)
        use_past = (
            past_txt
            and past_rel >= self.THRESH["past_relevance"]
            and self._cooldown["past"] <= 0.0
            and past_txt != self._last_used["past"]
            and not is_direct_question  # on évite de détourner une question courte
        )

        # 2) Candidat collocation aimée (faible proba, modulée par confiance)
        colloc_txt, colloc_rel = self._pick_collocation(ctx)
        p_try = self.THRESH["chance_colloc"] * (0.6 + 0.6 * conf)
        use_colloc = (
            colloc_txt
            and random.random() < p_try
            and colloc_rel >= self.THRESH["colloc_relevance"]
            and self._cooldown["colloc"] <= 0.0
            and colloc_txt != self._last_used["colloc"]
        )

        # Toujours max 1 ornement, priorité au passé s’il est pertinent
        arch = getattr(getattr(self.voice, "self_model", None), "arch", None)
        if arch and getattr(arch, "memory", None):
            try:
                arch.tactic_selector = getattr(arch, "tactic_selector", TacticSelector(arch))
                selector_ctx = ContextBuilder.build(
                    arch,
                    extra={
                        "pending_questions_count": len(
                            getattr(getattr(arch, "question_manager", None), "pending_questions", [])
                            or []
                        )
                    },
                )
                rule, why = arch.tactic_selector.pick(selector_ctx)
            except Exception:
                rule, why = (None, None)
            if rule:
                rule["last_used_ts"] = time.time()
                rule["usage_count"] = int(rule.get("usage_count", 0)) + 1
                if hasattr(arch.memory, "update_memory"):
                    arch.memory.update_memory(rule)
                else:
                    arch.memory.add_memory(rule)

                arch.memory.add_memory(
                    {
                        "kind": "decision_trace",
                        "rule_id": rule["id"],
                        "tactic": (rule.get("tactic") or {}).get("name"),
                        "ctx_snapshot": selector_ctx,
                        "why": why,
                        "ts": time.time(),
                    }
                )

                tac = (rule.get("tactic") or {}).get("name", "")
                if tac == "banter_leger" and "?" not in base:
                    base = base + " (clin d’œil)"
                elif tac == "ack_grateful":
                    base = base + " Merci, je le note."
                elif tac == "reformulation_empathique":
                    pass
                elif tac == "clarify_definition":
                    pass

        out = self._decorate_with_voice(base)
        if use_past:
            snippet = f"\n\n↪ En lien : {past_txt}"
            if len(snippet) <= budget:
                out += snippet
                self._cooldown["past"] = 2.0
                self._last_used["past"] = past_txt
                return out  # on s’arrête ici

        if use_colloc:
            snippet = f"\n\n({colloc_txt})"
            if len(snippet) <= budget:
                out += snippet
                self._cooldown["colloc"] = 2.0
                self._last_used["colloc"] = colloc_txt

        return out
