from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random

from AGI_Evolutive.core.structures.mai import MAI, new_mai, EvidenceRef, ImpactHypothesis
from AGI_Evolutive.knowledge.mechanism_store import MechanismStore

# Adapters (branchés sur tes modules v11)
# On suppose que interaction_miner renvoie des "patterns" normatifs déjà extraits.
# Chaque pattern contient: text, conditions (tokens/logiques), examples, strength
NormativePattern = Dict[str, Any]

class PrincipleInducer:
    """
    Transforme des motifs normatifs (lectures, dialogues, épisodes) en MAI candidats,
    puis délègue l'évaluation/ablation/promotion à ton pipeline (critic + world_model + self_improver).
    """
    def __init__(self, mechanism_store: MechanismStore):
        self.store = mechanism_store

    # ---- 1) Synthèse de MAI candidats depuis des patterns normatifs ----
    def induce_from_patterns(self, patterns: List[NormativePattern]) -> List[MAI]:
        """
        Patterns attendus (exemple minimal):
        {
          "text": "Garder une information confidentielle sans consentement explicite; sauf danger imminent.",
          "conditions": [
             {"op":"atom","name":"request_is_sensitive"},
             {"op":"atom","name":"audience_is_not_owner"},
             {"op":"not","args":[{"op":"atom","name":"has_consent"}]}
          ],
          "exceptions": [
             {"op":"atom","name":"imminent_harm_detected"}
          ],
          "suggested_actions": ["RefusePolitely","AskConsent","PartialReveal"],
          "evidence": [{"source":"doc:ethics.pdf#p12"}],
          "strength": 0.74
        }
        """
        candidates: List[MAI] = []
        for p in patterns:
            base_cond = {"op":"and","args": p.get("conditions", [])}
            exc = p.get("exceptions", [])
            if exc:
                # precondition := base_cond AND NOT(any exceptions)
                base_cond = {"op":"and","args":[ base_cond, {"op":"not","args":[{"op":"or","args":exc}]} ]}

            bids = [{"action_hint": a, "rationale": p.get("text","") } for a in p.get("suggested_actions", [])]

            evidence_refs = [EvidenceRef(**e) if isinstance(e, dict) else EvidenceRef(source=str(e))
                             for e in p.get("evidence", [])]

            m = new_mai(
                docstring=p.get("text","(principe appris)"),
                precondition_expr=base_cond,
                propose_spec={"bids": bids},
                evidence=evidence_refs,
                safety=["keep_explicit_promises"]  # invariant léger; l’identité peut en ajouter
            )
            # Hypothèse d'impact initiale grossière (affinée ensuite)
            m.expected_impact = ImpactHypothesis(trust_delta=+0.2, harm_delta=-0.2, identity_coherence_delta=+0.1, uncertainty=0.5)
            candidates.append(m)
        return candidates

    # ---- 2) Pré-évaluation rapide (heuristique) AVANT sandbox lourde ----
    def prefilter(self, mai_list: List[MAI], history_stats: Dict[str, float]) -> List[MAI]:
        # Exemple trivial: limite le nombre de canary à promouvoir (taux d’exploration global)
        max_canary = int(max(1, history_stats.get("mai_max_canary", 3)))
        random.shuffle(mai_list)
        return mai_list[:max_canary]

    # ---- 3) Remise au pipeline d’évaluation existant ----
    def submit_for_evaluation(self, mai_list: List[MAI]) -> None:
        """
        Ici, on NE ré-implémente PAS critic/world_model/self_improver.
        On les appelle : contrefactuels + social_critic + ablation + promote → store.add/update.
        Esquisse d’intégration (à adapter à tes signatures internes):
        """
        # PSEUDOCODE (à mapper à ton evolution_manager / self_improver)
        # for m in mai_list:
        #   ok = evolution_manager.evaluate_mechanism(m)
        #   if ok: self.store.add(m)
        pass
