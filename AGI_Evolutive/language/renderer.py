from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional

import random
import re
import time

from AGI_Evolutive.social.tactic_selector import TacticSelector
from AGI_Evolutive.social.interaction_rule import ContextBuilder
from AGI_Evolutive.core.structures.mai import MAI

from .nlg import NLGContext, apply_mai_bids_to_nlg, paraphrase_light, join_tokens
from .style_critic import StyleCritic


def _tokens(s: str) -> set:
    return set(re.findall(r"[A-Za-zÀ-ÿ]{3,}", (s or "").lower()))


def _build_language_state_snapshot(arch, ctx: Dict[str, Any]) -> Dict[str, Any]:
    dialogue_ctx = ctx.get("dialogue") or ctx.get("dialogue_state")
    if dialogue_ctx is None:
        dialogue_ctx = getattr(getattr(arch, "language", None), "state", None)
    return {
        "beliefs": getattr(arch, "beliefs", None),
        "self_model": getattr(arch, "self_model", None),
        "dialogue": dialogue_ctx,
        "world": getattr(arch, "world_model", None),
    }


def _apply_action_hint(text: str, hint: str) -> str:
    hint = (hint or "").strip()
    lower = text.lower()
    if hint == "AskConsent":
        prefix = "Avant de poursuivre, pourrais-tu confirmer que je peux partager ces informations ? "
        if not lower.startswith(prefix.lower()):
            return prefix + text
    elif hint == "RefusePolitely":
        apology = "Je suis désolé, je ne peux pas partager cette information."
        if apology.lower() not in lower:
            return f"{apology} {text}".strip()
    elif hint == "PartialReveal":
        note = "(Je partage uniquement ce qui est approprié pour protéger la confidentialité.)"
        if note.lower() not in lower:
            return f"{text}\n\n{note}"
    elif hint == "RephraseRespectfully":
        marker = "Je vais reformuler avec plus de délicatesse :"
        if marker.lower() not in lower:
            return f"{marker} {text}"
    return text


class LanguageRenderer:
    def __init__(self, voice_profile, lexicon, ranker=None):
        self.voice = voice_profile
        self.lex = lexicon
        self.ranker = ranker
        # anti-spam / fréquence
        self._cooldown = {"past": 0.0, "colloc": 0.0}
        self._last_used = {"past": "", "colloc": ""}

        # seuils réglables
        self.THRESH = {
            "past_relevance": 0.25,   # pertinence mini lien passé
            "colloc_relevance": 0.20, # pertinence mini collocation
            "conf_min": 0.55,         # confiance mini pour ornement
            "chance_colloc": 0.35,    # proba de tenter une collocation (modulée)
            "quote_prob": 0.35,
        }

        self.critic = StyleCritic(max_chars=1200)
        self._rand = random.Random()

    def rand(self) -> float:
        return self._rand.random()

    # ---------- utilitaires ----------
    def apply_action_hint(self, text: str, hint: str) -> str:
        """Expose l'utilitaire d'application d'hints MAI pour d'autres pipelines."""

        return _apply_action_hint(text, hint)

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

    def _decrease_cooldowns(self, store: Optional[Dict[str, float]] = None) -> None:
        # refroidit un peu à chaque appel
        target = store if store is not None else self._cooldown
        for k in target:
            target[k] = max(0.0, target[k] - 0.34)

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

    def render_reply(self, semantics: Dict[str, Any], ctx: Dict[str, Any], *, dry_run: bool = False) -> str:
        """
        Règle : on n’ajoute un lien au passé / une collocation QUE si :
        - confiance >= conf_min
        - pertinence >= seuils
        - budget > 0
        - cooldown OK et pas de doublon
        Et au plus 1 ornement par réponse (priorité au lien passé).
        """
        ctx = ctx or {}
        if dry_run:
            ctx = dict(ctx)

        cooldown = dict(self._cooldown)
        last_used = dict(self._last_used)

        self._decrease_cooldowns(cooldown)
        base = (semantics.get("text") or "").strip() or "Je te réponds en tenant compte de notre historique."
        arch = getattr(getattr(self.voice, "self_model", None), "arch", None)
        policy = getattr(arch, "policy", None) if arch else None
        state_snapshot: Dict[str, Any] = {}
        predicate_registry: Dict[str, Any] = {}
        applicable_mais: List[MAI] = []
        if arch and policy and hasattr(policy, "build_predicate_registry"):
            try:
                state_snapshot = ctx.get("state_snapshot") or _build_language_state_snapshot(arch, ctx)
                predicate_registry = policy.build_predicate_registry(state_snapshot)
            except Exception:
                state_snapshot = {}
                predicate_registry = {}

        nlg_context = NLGContext(base, _apply_action_hint)
        try:
            applicable_mais = apply_mai_bids_to_nlg(nlg_context, state_snapshot, predicate_registry)
        except Exception:
            applicable_mais = []
        base = nlg_context.text
        applied_hints = nlg_context.applied_hints()
        if applied_hints and not dry_run:
            ctx.setdefault("applied_action_hints", []).extend(applied_hints)
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
            and cooldown["past"] <= 0.0
            and past_txt != last_used["past"]
            and not is_direct_question  # on évite de détourner une question courte
        )

        # 2) Candidat collocation aimée (faible proba, modulée par confiance)
        colloc_txt, colloc_rel = self._pick_collocation(ctx)
        p_try = self.THRESH["chance_colloc"] * (0.6 + 0.6 * conf)
        use_colloc = (
            colloc_txt
            and random.random() < p_try
            and colloc_rel >= self.THRESH["colloc_relevance"]
            and cooldown["colloc"] <= 0.0
            and colloc_txt != last_used["colloc"]
        )

        # Toujours max 1 ornement, priorité au passé s’il est pertinent
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
                if not dry_run:
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
                cooldown["past"] = 2.0
                last_used["past"] = past_txt
                if not dry_run:
                    self._cooldown = cooldown
                    self._last_used = last_used
                return out  # on s’arrête ici

        if use_colloc:
            snippet = f"\n\n({colloc_txt})"
            if len(snippet) <= budget:
                out += snippet
                cooldown["colloc"] = 2.0
                last_used["colloc"] = colloc_txt
                if not dry_run:
                    self._cooldown = cooldown
                    self._last_used = last_used

        if not dry_run and ctx.get("omitted_content") and arch:
            payload = {
                "reason": "MAI-driven",
                "mai_ids": [mai.id for mai in applicable_mais],
                "evidence": [
                    asdict(doc)
                    for mai in applicable_mais
                    for doc in getattr(mai, "provenance_docs", [])
                ],
            }
            audit = getattr(arch, "audit", None)
            if audit and hasattr(audit, "log_omission_justifiee"):
                try:
                    audit.log_omission_justifiee(**payload)
                except Exception:
                    pass
            else:
                logger = getattr(arch, "logger", None)
                if logger and hasattr(logger, "write"):
                    try:
                        logger.write("nlg.omission", **payload)
                    except Exception:
                        pass

        if not dry_run:
            self._cooldown = cooldown
            self._last_used = last_used
        return out

    # --- Génère K variantes (macros + paraphrases légères) ---
    def render_reply_candidates(self, ctx: Dict[str, Any], base_plan: Dict[str, Any], K: int = 4) -> List[str]:
        plan = dict(base_plan or {})
        ctx = dict(ctx or {})

        neutral = self._render_once(ctx, plan, macro=None)
        candidates: List[str] = [neutral]

        macros = ["taquin", "coach", "sobre", "deadpan"]
        self._rand.shuffle(macros)
        for macro in macros:
            if len(candidates) >= max(1, K):
                break
            variant = self._render_once(ctx, plan, macro=macro)
            if variant and variant not in candidates:
                candidates.append(variant)

        out: List[str] = []
        for i, cand in enumerate(candidates):
            if i == 0:
                out.append(cand)
            else:
                out.append(paraphrase_light(cand, prob=0.30))
        return out[: max(1, K)]

    def _render_once(self, ctx: Dict[str, Any], plan: Dict[str, Any], macro: Optional[str] = None) -> str:
        sp = getattr(self.voice, "style_policy", None)
        snapshot = None
        mode_snapshot = None
        macro = (macro or "").lower() or None
        if sp and macro and hasattr(sp, "apply_macro"):
            try:
                snapshot = dict(getattr(sp, "params", {}))
                mode_snapshot = getattr(sp, "current_mode", None)
                sp.apply_macro(macro)
            except Exception:
                snapshot = None
                mode_snapshot = None

        try:
            text = self.render_reply(plan, ctx, dry_run=True)
        finally:
            if sp and snapshot is not None:
                try:
                    sp.params.update(snapshot)
                    if mode_snapshot is not None:
                        sp.current_mode = mode_snapshot
                except Exception:
                    pass

        macro_text = text
        if macro:
            if macro == "taquin" and "😉" not in macro_text:
                macro_text = macro_text.strip() + " 😉"
            elif macro == "coach":
                prefix = "Coach mode ▶ "
                if not macro_text.lower().startswith(prefix.lower()):
                    macro_text = f"{prefix}{macro_text}" if macro_text else prefix.rstrip()
            elif macro == "sobre":
                macro_text = re.sub(r"[🙂😉😊]+\s*", "", macro_text).strip()
            elif macro == "deadpan":
                lines: List[str] = []
                for line in macro_text.splitlines():
                    normalized = line.replace("!", ".").replace("…", ".")
                    tokens = normalized.split()
                    lines.append(join_tokens(tokens))
                macro_text = "\n".join(lines).strip()

        return macro_text.strip()

    def render_final(self, ctx: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        ctx_local = dict(ctx or {})
        plan_local = dict(plan or {})
        candidates = self.render_reply_candidates(ctx_local, plan_local, K=4)

        local_ctx = {
            "style": dict(getattr(getattr(self.voice, "style_policy", None), "params", {}))
        }
        scored: List[Tuple[float, int, str]] = []
        for idx, candidate in enumerate(candidates):
            try:
                score = (
                    float(self.ranker.score(local_ctx, candidate))
                    if self.ranker is not None
                    else 0.5
                )
            except Exception:
                score = 0.5
            scored.append((score, idx, candidate))

        if not scored:
            return {"text": "", "chosen": None, "alts": []}

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, chosen_idx, best_text = scored[0]

        analysis = self.critic.analyze(best_text)
        rewritten = self.critic.rewrite(best_text)
        best_text = rewritten

        final_text, quote_meta = self._maybe_add_quote(best_text, ctx_local)

        return {
            "text": final_text,
            "chosen": {
                "idx": chosen_idx,
                "score": best_score,
                "text": final_text,
                "analysis": analysis,
                "quote": quote_meta,
            },
            "alts": [
                {"idx": idx, "score": score, "text": text}
                for score, idx, text in scored[1:]
            ],
        }

    def _ctx_is_safe_for_aside(self, ctx: Dict[str, Any]) -> bool:
        user_msg = (ctx.get("last_user_msg") or ctx.get("last_message") or "").lower()
        for bad in ["décès", "urgence", "mauvaise nouvelle", "licenciement", "plainte"]:
            if bad in user_msg:
                return False
        return True

    def _maybe_add_quote(self, text: str, ctx: Dict[str, Any]):
        quote_meta = None
        qm = getattr(self.voice, "quote_memory", None) or getattr(self, "quote_memory", None)
        if not qm:
            return text, quote_meta
        if (
            len(text) < 900
            and self._ctx_is_safe_for_aside(ctx)
            and self.rand() < self.THRESH["quote_prob"]
        ):
            try:
                context_seed = text + " " + (ctx.get("last_user_msg") or "")
                quote = qm.sample(context=context_seed)
            except Exception:
                quote = None
            if quote:
                quote_used = quote
                if len(quote.split()) > 20:
                    quote_used = paraphrase_light(quote, prob=0.45)
                text = f"{text}\n\n(Clin d’œil) {quote_used}".strip()
                quote_meta = {"len": len(quote_used), "raw_len": len(quote)}
        return text, quote_meta
