# AGI_Evolutive/social/interaction_miner.py
# Induction de règles sociales ⟨Contexte→Tactique→Effets_attendus⟩
# depuis des dialogues inbox. Zéro LLM, heuristiques + ontologie si dispo.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import hashlib
import math

from AGI_Evolutive.social.interaction_rule import (
    InteractionRule, Predicate, TacticSpec, ContextBuilder
)

# ---------------------- utilitaires ----------------------
def _now() -> float: return time.time()
def _hash(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))

# Détections rapides FR (tu peux enrichir sans rien casser)
RE_QMARK = re.compile(r"\?\s*$")
RE_COMPLIMENT = re.compile(r"\b(bravo|bien joué|trop fort|j'adore|excellent|génial|super)\b", re.I)
RE_THANKS     = re.compile(r"\b(merci|thanks|thx)\b", re.I)
RE_DISAGREE   = re.compile(r"\b(je ne suis pas d'accord|pas d'accord|non|je pense pas|pas sûr|bof)\b", re.I)
RE_EXPLAIN    = re.compile(r"\b(parce que|car|en fait|la raison|explication|voilà pourquoi)\b", re.I)
RE_CONFUSED   = re.compile(r"\b(je ne comprends pas|c'est quoi|hein\??|pardon\??|qu'est-ce que|explique)\b", re.I)
RE_CLARIFY    = re.compile(r"\b(alors|donc|autrement dit|en clair|pour être clair|ça veut dire)\b", re.I)
RE_INSINUATE  = re.compile(r"\b(si tu le dis|bien sûr+|okayyy+|hmm+|hum+|lol+|mdr+|ouais c'est ça)\b", re.I)
RE_ACK        = re.compile(r"\b(ok|d'accord|noté|je vois|ça marche|compris)\b", re.I)

# ---------------------- structures ----------------------
@dataclass
class DialogueTurn:
    speaker: str
    text: str
    act: Optional[str] = None
    valence: float = 0.0

# ---------------------- InteractionMiner ----------------------
class InteractionMiner:
    """
    Lit du texte de conversation H↔H et fabrique des InteractionRule
    basées sur paires adjacentes + indices d'implicature.
    S'appuie sur Ontology/Beliefs/analyzers si dispo, sinon heuristiques.
    """
    def __init__(self, arch):
        self.arch = arch

    # ----------- API publique -----------
    def mine_file(self, path: str) -> List[InteractionRule]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return []
        return self.mine_text(text, source=f"inbox:{path}")

    def mine_text(self, text: str, source: str = "inbox:unknown") -> List[InteractionRule]:
        turns = self._parse_turns(text)
        self._annotate_acts(turns)            # remplit .act (heuristique + analyzers si dispo)
        rules = self._extract_rules(turns, source)
        rules = self._merge_duplicates(rules) # fusionne mêmes règles avec +evidence
        return rules

    # ----------- Auto-évaluation & simulation -----------
    def schedule_self_evaluation(self, ctx, payload: Dict[str, Any]):
        """Background job entry point used by DocumentIngest."""
        rule_dict = payload.get("rule") if isinstance(payload, dict) else None
        arch = payload.get("arch") if isinstance(payload, dict) else None
        if rule_dict is None:
            return {"status": "skipped", "reason": "no_rule"}
        try:
            rule = InteractionRule.from_dict(rule_dict)
        except Exception:
            return {"status": "skipped", "reason": "invalid_rule"}
        arch = arch or self.arch
        outcome = self._simulate_rule(rule)
        self._persist_evaluation(rule, outcome, arch)
        return outcome

    def _simulate_rule(self, rule: InteractionRule) -> Dict[str, Any]:
        """Cheap multi-factor simulation -> returns a pseudo-outcome payload."""
        now = time.time()
        conf = float(getattr(rule, "confidence", 0.5) or 0.5)
        support = len(getattr(rule, "evidence", []) or [])
        predicate_weight = sum(float(getattr(pred, "confidence", 0.6) or 0.6) for pred in rule.predicates)
        predicate_weight = predicate_weight / max(1, len(rule.predicates)) if rule.predicates else 0.5
        # heuristique: plus de support & predicates cohérents => meilleur score
        base = 0.45 + (conf * 0.35) + (predicate_weight * 0.2)
        support_boost = math.tanh(support / 4.0) * 0.1
        score = max(0.0, min(1.0, base + support_boost))
        risk = max(0.0, 1.0 - score)
        action = "deploy" if score >= 0.62 else ("review" if score >= 0.5 else "hold")
        counterexamples: List[Dict[str, Any]] = []
        if action != "deploy":
            counterexamples.append({
                "rule_id": rule.id,
                "reason": "score_below_threshold",
                "score": score,
                "ts": now,
            })
        outcome = {
            "rule_id": rule.id,
            "score": round(score, 3),
            "risk": round(risk, 3),
            "action": action,
            "support": support,
            "timestamp": now,
            "counterexamples": counterexamples,
        }
        return outcome

    def _persist_evaluation(self, rule: InteractionRule, outcome: Dict[str, Any], arch) -> None:
        mem = getattr(arch, "memory", None)
        if not mem or not hasattr(mem, "add_memory"):
            return
        try:
            payload = {
                "kind": "interaction_rule_evaluation",
                "rule_id": rule.id,
                "outcome": outcome,
                "ts": outcome.get("timestamp", time.time()),
            }
            mem.add_memory(payload)
        except Exception:
            pass
        try:
            if outcome.get("counterexamples"):
                for ce in outcome["counterexamples"]:
                    mem.add_memory({
                        "kind": "interaction_counterexample",
                        "rule_id": rule.id,
                        "details": ce,
                    })
        except Exception:
            pass
        try:
            rule_dict = rule.to_dict()
            rule_dict.setdefault("evidence", {})["last_review_ts"] = outcome.get("timestamp", time.time())
            rule_dict["evidence"]["auto_score"] = outcome.get("score")
            rule_dict["evidence"]["recommended_action"] = outcome.get("action")
            if hasattr(mem, "update_memory"):
                mem.update_memory(rule_dict)
            else:
                mem.add_memory(rule_dict)
        except Exception:
            pass

    # ----------- Parsing basique des tours ----------
    def _parse_turns(self, text: str) -> List[DialogueTurn]:
        """
        Heuristique tolérante: détecte 'A:', 'B:', 'User:', '—', ou lignes sèches.
        On essaye de conserver l'alternance; sinon on attribue speakers génériques.
        """
        lines = [l.strip() for l in re.split(r"[\r\n]+", text or "") if l.strip()]
        turns: List[DialogueTurn] = []
        cur_speaker = "A"
        for ln in lines:
            m = re.match(r"^([A-Za-z]{1,12})\s*[:\-]\s*(.+)$", ln)
            if m:
                spk, utt = m.group(1), m.group(2)
                turns.append(DialogueTurn(spk.strip(), utt.strip()))
                cur_speaker = spk.strip()
            else:
                # bullets/quote style
                if ln.startswith(("-", "•", "—", ">")):
                    ln = ln.lstrip("-•—> ").strip()
                turns.append(DialogueTurn(cur_speaker, ln))
                cur_speaker = "B" if cur_speaker == "A" else "A"
        return turns

    # ----------- Annotation d'actes de dialogue ----------
    def _annotate_acts(self, turns: List[DialogueTurn]) -> None:
        """
        Si un analyzer existe (arch.conversation_classifier, arch.analyzers), on l'utilise.
        Sinon heuristiques simples.
        """
        # Essayez analyzers si dispo
        used_external = False
        try:
            clf = getattr(self.arch, "conversation_classifier", None)
            if clf and hasattr(clf, "predict_acts"):
                acts = clf.predict_acts([t.text for t in turns])
                for t, a in zip(turns, acts):
                    t.act = a
                used_external = True
        except Exception:
            pass

        if used_external:
            return

        # Heuristique locale
        for t in turns:
            low = t.text.lower()
            if RE_QMARK.search(low):
                t.act = t.act or "question"
            elif RE_COMPLIMENT.search(low):
                t.act = t.act or "compliment"
            elif RE_DISAGREE.search(low):
                t.act = t.act or "disagreement"
            elif RE_CONFUSED.search(low):
                t.act = t.act or "confusion"
            elif RE_INSINUATE.search(low):
                t.act = t.act or "insinuation"
            elif RE_THANKS.search(low):
                t.act = t.act or "thanks"
            elif RE_ACK.search(low):
                t.act = t.act or "ack"
            elif RE_EXPLAIN.search(low):
                t.act = t.act or "explain"
            elif RE_CLARIFY.search(low):
                t.act = t.act or "clarify"
            else:
                t.act = t.act or "statement"

    # ----------- Extraction de règles (paires + implicatures) ----------
    def _extract_rules(self, turns: List[DialogueTurn], source: str) -> List[InteractionRule]:
        rules: List[InteractionRule] = []
        n = len(turns)
        for i in range(n - 1):
            a, b = turns[i], turns[i + 1]
            pair = (a.act, b.act)

            # 1) Question -> Answer concise
            if a.act == "question" and b.act not in ("question", None):
                rules.append(self._rule_from_pair(
                    kind="question_answer",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "question", 1.0),
                        Predicate("risk_level",   "in", ["low","medium"], 0.6),
                        Predicate("persona_alignment", "ge", 0.2, 0.4),
                    ],
                    tactic=TacticSpec("answer_concise", {"max_len": 200, "ensure_ack": True}),
                    base_conf=0.68
                ))

            # 2) Compliment -> Acknowledge grateful
            if RE_COMPLIMENT.search(a.text) and (b.act in ("thanks","ack","statement")):
                rules.append(self._rule_from_pair(
                    kind="compliment_thanks",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "compliment", 1.1),
                        Predicate("risk_level",   "in", ["low","medium"], 0.7),
                    ],
                    tactic=TacticSpec("ack_grateful", {"warmth": "auto"}),
                    base_conf=0.70
                ))

            # 3) Insinuation -> Banter léger ou Reformulation empathique (selon risk)
            if a.act == "insinuation" and (b.act in ("ack","statement","insinuation")):
                # On crée DEUX règles candidates (policy/selector trancheront selon context)
                rules.append(self._rule_from_pair(
                    kind="insinuation_banter",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "insinuation", 1.2),
                        Predicate("risk_level",   "in", ["low","medium"], 0.8),
                        Predicate("persona_alignment", "ge", 0.3, 0.6),
                    ],
                    tactic=TacticSpec("banter_leger", {"soft": True, "max_len_delta": 40}),
                    base_conf=0.64
                ))
                rules.append(self._rule_from_pair(
                    kind="insinuation_reformulate",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "insinuation", 1.2),
                        Predicate("risk_level",   "in", ["medium","high"], 0.9),
                    ],
                    tactic=TacticSpec("reformulation_empathique", {"mirror_ratio": 0.4}),
                    base_conf=0.62
                ))

            # 4) Disagreement -> Explain/Clarify
            if a.act == "disagreement" and (b.act in ("explain","clarify","statement")):
                rules.append(self._rule_from_pair(
                    kind="disagree_explain",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "disagreement", 1.0),
                        Predicate("risk_level",   "in", ["low","medium","high"], 0.7),
                    ],
                    tactic=TacticSpec("clarification_rationale", {"connective": "parce que"}),
                    base_conf=0.66
                ))

            # 5) Confusion -> Clarify / Reformulation
            if a.act == "confusion" and (b.act in ("clarify","explain","statement")):
                rules.append(self._rule_from_pair(
                    kind="confusion_clarify",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "confusion", 1.0),
                        Predicate("risk_level",   "in", ["low","medium"], 0.8),
                    ],
                    tactic=TacticSpec("clarify_definition", {"ensure_example": True}),
                    base_conf=0.69
                ))
                rules.append(self._rule_from_pair(
                    kind="confusion_reformulation",
                    a=a, b=b, source=source,
                    preds=[
                        Predicate("dialogue_act", "eq", "confusion", 1.0),
                        Predicate("risk_level",   "in", ["low","medium","high"], 0.7),
                        Predicate("persona_alignment", "ge", 0.2, 0.4),
                    ],
                    tactic=TacticSpec("reformulation_empathique", {"mirror_ratio": 0.6}),
                    base_conf=0.65
                ))

        # enrichissement par ontologie (synonymes -> implicature_hint)
        for r in rules:
            try:
                onto = getattr(self.arch, "ontology", None)
                if not onto: 
                    continue
                # Exemple: si act=insinuation, ajoute un prédicat implicature_hint ∈ {ironie, sous-entendu}
                if any(p.op == "eq" and p.key == "dialogue_act" and p.value == "insinuation"
                       for p in r.context_predicates):
                    r.context_predicates.append(Predicate("implicature_hint", "in", ["sous-entendu","ironie"], 0.4))
            except Exception:
                pass

        return rules

    def _rule_from_pair(self, kind: str, a: DialogueTurn, b: DialogueTurn, source: str,
                        preds: List[Predicate], tactic: TacticSpec, base_conf: float) -> InteractionRule:
        r = InteractionRule.build(preds, tactic, provenance={
            "source": source, "kind": kind,
            "evidence": {"a": a.text, "b": b.text}
        })
        # amorce de confiance
        r.confidence = clamp(base_conf, 0.0, 1.0)
        # tags utiles
        r.tags = list(set((r.tags or []) + ["mined","inbox"]))
        return r

    # ----------- fusion de doublons / agrégation d'évidence ----------
    def _merge_duplicates(self, rules: List[InteractionRule]) -> List[InteractionRule]:
        merged: Dict[str, InteractionRule] = {}
        for r in rules:
            if r.id in merged:
                # on augmente légèrement la confiance si plusieurs occurrences
                merged[r.id].confidence = clamp(0.6*merged[r.id].confidence + 0.4*r.confidence, 0.0, 1.0)
                # concat (sans explosion)
                ev = merged[r.id].provenance.get("evidence", {})
                if isinstance(ev, dict) and isinstance(r.provenance.get("evidence"), dict):
                    # garde 3 exemples max
                    ex = []
                    if isinstance(ev.get("a"), str) and isinstance(ev.get("b"), str):
                        ex.append(ev)
                    ex.append(r.provenance["evidence"])
                    merged[r.id].provenance["evidence_multi"] = (merged[r.id].provenance.get("evidence_multi", []) + ex)[-3:]
            else:
                merged[r.id] = r
        return list(merged.values())
