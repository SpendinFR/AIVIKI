# Gestion de l'autonomie : auto-seed d'objectifs, micro-constitution, agenda, déduplication et fallback
# Compatible avec l'architecture existante (GoalSystem, Metacognition, Memory, Perception, Language...)
# Aucune dépendance externe (stdlib uniquement). Logs lisibles dans ./logs/autonomy.log

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
import os, time, uuid, json, threading

# --------- Structures ---------

@dataclass
class AgendaItem:
    id: str
    title: str
    rationale: str
    kind: str              # "learning" | "reasoning" | "intake" | "alignment" | "meta"
    priority: float        # 0..1
    created_at: float
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued" # queued | running | done | skipped
    dedupe_key: Optional[str] = None

# --------- Autonomy Manager ---------

class AutonomyManager:
    """
    Autonomie de l'agent :
      - micro-constitution (principes) -> alignement doux
      - auto-seed d'objectifs à partir de l'état interne + environnement (inbox)
      - gestion d'agenda (priorités, déduplication, fallback si vide)
      - intégration souple avec GoalSystem (si disponible) + logs
    """
    def __init__(self,
                 architecture,
                 goal_system=None,
                 metacognition=None,
                 memory=None,
                 perception=None,
                 language=None):

        self.arch = architecture
        self.goals = goal_system
        self.metacog = metacognition
        self.memory = memory
        self.perception = perception
        self.language = language

        # Flags / Config
        self.SELF_SEED: bool = True              # auto-génération par défaut
        self.FALLBACK_AFTER_TICKS: int = 8       # si rien d’utile émis → fallback
        self.MAX_QUEUE: int = 50
        self.MIN_USEFUL_QUESTIONS: int = 1       # toujours pousser un minimum de questions utiles
        self.LAST_N_DEDUPE: int = 40             # fenêtre de déduplication

        # Micro-constitution : principes (pas une todo-list)
        self.constitution: List[str] = [
            "Toujours expliciter ce qui manque (données, contraintes) avant d’agir.",
            "Optimiser le ratio progrès/coût (temps, confusion, dette).",
            "Améliorer en priorité les capacités générales (langage, raisonner, apprendre).",
            "Valider par boucles courtes: hypothèses → preuves/feedback.",
            "Respecter l’humain (clarté, coopération, sécurité)."
        ]

        # Fallback seed (au cas où l’auto-seed n’émet rien d’utile)
        self.fallback_seed: List[Dict[str, Any]] = [
            {
                "title": "Cartographier mes modules et leurs métriques",
                "kind": "meta",
                "priority": 0.9,
                "rationale": "Avoir une vue claire pour décider quoi améliorer en premier.",
                "payload": {"action": "snapshot_modules"}
            },
            {
                "title": "Analyser l’inbox et créer un plan d’intégration",
                "kind": "intake",
                "priority": 0.8,
                "rationale": "L’environnement est source de contexte et d’apprentissage.",
                "payload": {"action": "scan_inbox", "path": "./inbox"}
            },
            {
                "title": "Améliorer ma compréhension du langage (glossaire perso)",
                "kind": "learning",
                "priority": 0.75,
                "rationale": "Meilleure compréhension → meilleures interactions.",
                "payload": {"action": "build_glossary", "target": "core_terms"}
            }
        ]

        # État interne
        self.agenda: deque[AgendaItem] = deque(maxlen=self.MAX_QUEUE)
        self.recent_keys: deque[str] = deque(maxlen=self.LAST_N_DEDUPE)
        self.ticks_without_useful: int = 0
        self.last_tick = 0.0
        self._lock = threading.Lock()

        # Journal
        self.log_dir = "./logs"
        self.log_path = os.path.join(self.log_dir, "autonomy.log")
        os.makedirs(self.log_dir, exist_ok=True)
        self._log("🔧 AutonomyManager prêt (SELF_SEED=True, fallback activé)")

    # ---------- Public API ----------

    def tick(self) -> None:
        """
        Appeler à chaque cycle (ex: dans CognitiveArchitecture.cycle()).
        - Sème si nécessaire (auto-seed)
        - Émet au moins une question utile si contexte flou
        - Exécute (légèrement) certaines tâches “automatiques” (scan inbox, snapshot…)
        - Pousse les objectifs vers GoalSystem si présent
        """
        with self._lock:
            now = time.time()
            if now - self.last_tick < 0.5:
                return  # évite le spam si le cycle est très rapide
            self.last_tick = now

            # 1) Sème de nouveaux objectifs si l’agenda est pauvre
            self._maybe_seed()

            # 2) Évite la stagnation : s’il n’y a pas d’élément “utile”, fallback
            if self._agenda_is_poor():
                self._log("⚠️ Agenda peu utile → fallback seed")
                self._inject_fallback_seed()

            # 3) Émet au moins une question utile si besoin
            self._maybe_emit_useful_question()

            # 4) Essaie de “démarrer” la prochaine tâche exécutable (automatique)
            item = self._pop_next_item()
            if item:
                self._execute_item(item)

    # ---------- Seeding ----------

    def _maybe_seed(self) -> None:
        if not self.SELF_SEED:
            return
        proposals = self._auto_seed_proposals()
        added = 0
        for p in proposals:
            if self._push_if_new(p):
                added += 1
        if added:
            self._log(f"🌱 Auto-seed: +{added} objectif(s)")

    def _auto_seed_proposals(self) -> List[Dict[str, Any]]:
        """
        Génère des propositions à partir :
          - des métriques faibles (metacognition.performance_tracking)
          - de la présence de fichiers en inbox
          - de lacunes de langage (si language présent)
        """
        props: List[Dict[str, Any]] = []

        # a) lacunes / signaux faibles depuis la métacognition
        weak = self._detect_weak_capabilities()
        for cap, score in weak:
            props.append({
                "title": f"Améliorer la capacité « {cap} »",
                "kind": "learning",
                "priority": 0.7 + (0.15 * (1.0 - score)),
                "rationale": f"La métrique « {cap} » est faible ({score:.2f}).",
                "payload": {"action": "improve_metric", "metric": cap}
            })

        # b) environnement (inbox)
        inbox_path = "./inbox"
        if os.path.isdir(inbox_path) and self._dir_has_content(inbox_path):
            props.append({
                "title": "Analyser l’inbox (fichiers récents)",
                "kind": "intake",
                "priority": 0.8,
                "rationale": "Nouveaux indices contextuels disponibles.",
                "payload": {"action": "scan_inbox", "path": inbox_path}
            })

        # c) langage / explication — toujours utile si pas de base lexicale
        if self.language and hasattr(self.language, "known_terms"):
            if len(getattr(self.language, "known_terms", {})) < 20:
                props.append({
                    "title": "Construire un glossaire minimal",
                    "kind": "learning",
                    "priority": 0.7,
                    "rationale": "Renforcer la base sémantique (termes fréquents).",
                    "payload": {"action": "build_glossary", "target": "core_terms"}
                })
        else:
            # si module language inconnu → tâche d’investigation
            props.append({
                "title": "Évaluer mes capacités de langage",
                "kind": "meta",
                "priority": 0.65,
                "rationale": "Identifier mes limites de compréhension/production.",
                "payload": {"action": "self_language_probe"}
            })

        # d) principe : toujours demander ce qui manque si le contexte est flou
        if self._context_is_fuzzy():
            props.append({
                "title": "Clarifier le contexte et les contraintes",
                "kind": "alignment",
                "priority": 0.85,
                "rationale": "Constitution: expliciter ce qui manque avant d’agir.",
                "payload": {"action": "ask_user", "question": self._build_clarifying_question()}
            })

        return props

    # ---------- Exécution locale (légère) ----------

    def _execute_item(self, item: AgendaItem) -> None:
        """Exécute rapidement les tâches simples; sinon pousse vers GoalSystem."""
        item.status = "running"
        self._log(f"▶️ Exécution: {item.title} [{item.kind}]")

        action = (item.payload or {}).get("action")

        try:
            if action == "scan_inbox":
                listed = self._list_inbox(item.payload.get("path", "./inbox"))
                self._log(f"📂 Inbox: {len(listed)} élément(s) détecté(s).")
                # Ajoute sous-tâches d’intégration
                for name in listed[:20]:
                    self._push_if_new({
                        "title": f"Intégrer le fichier « {name} »",
                        "kind": "intake",
                        "priority": 0.6,
                        "rationale": "Transformer le contenu en connaissance exploitable.",
                        "payload": {"action": "ingest_file", "filename": name}
                    })

            elif action == "snapshot_modules":
                snap = self._snapshot_modules()
                self._write_json("./logs/autonomy_snapshot.json", snap)
                self._log("🧭 Snapshot des modules écrit dans logs/autonomy_snapshot.json")

            elif action == "build_glossary":
                # On ne modifie pas le code du module langage; on prépare juste une todo structurée.
                terms = self._propose_core_terms()
                self._write_json("./logs/proposed_glossary.json", {"terms": terms})
                self._log("🗂️ Glossaire proposé dans logs/proposed_glossary.json")

            elif action == "self_language_probe":
                report = self._language_probe()
                self._write_json("./logs/language_probe.json", report)
                self._log("🔎 Rapport de sonde langage dans logs/language_probe.json")

            elif action == "ask_user":
                q = item.payload.get("question") or "De quoi as-tu besoin que je fasse en priorité ?"
                print(f"\n🤔 (Autonomy) Question: {q}\n")
                # rien d’autre à faire; la réponse utilisateur alimente la suite

            else:
                # Si ce n'est pas une tâche locale → pousser vers GoalSystem si dispo
                self._push_to_goal_system(item)

        except Exception as e:
            self._log(f"❌ Erreur exécution tâche: {e}")

        item.status = "done"

    def _push_to_goal_system(self, item: AgendaItem) -> None:
        if not self.goals:
            return
        # on tente des API communes sans casser si absentes
        pushed = False
        try:
            if hasattr(self.goals, "add_goal"):
                self.goals.add_goal({
                    "id": item.id,
                    "title": item.title,
                    "rationale": item.rationale,
                    "kind": item.kind,
                    "priority": item.priority,
                    "payload": item.payload
                })
                pushed = True
            elif hasattr(self.goals, "register_goal"):
                self.goals.register_goal(item.title, item.payload)
                pushed = True
        except Exception as e:
            self._log(f"⚠️ GoalSystem indisponible: {e}")

        if pushed:
            self._log(f"📌 Objectif poussé vers GoalSystem: {item.title}")

    # ---------- Utilitaires d’agenda ----------

    def _push_if_new(self, p: Dict[str, Any]) -> bool:
        """Ajoute un item si pas de doublon récent (dedupe_key)."""
        dedupe_key = p.get("dedupe_key") or f"{p.get('kind')}::{p.get('title')}"
        if dedupe_key in self.recent_keys:
            return False

        itm = AgendaItem(
            id=str(uuid.uuid4()),
            title=p["title"],
            rationale=p.get("rationale", ""),
            kind=p.get("kind", "meta"),
            priority=float(p.get("priority", 0.5)),
            created_at=time.time(),
            payload=p.get("payload", {}),
            status="queued",
            dedupe_key=dedupe_key
        )
        self.agenda.append(itm)
        self.recent_keys.append(dedupe_key)
        return True

    def _pop_next_item(self) -> Optional[AgendaItem]:
        if not self.agenda:
            return None
        # priorité simple (max priority, plus ancien en cas d’égalité)
        best_idx = None
        best_score = -1.0
        for i, itm in enumerate(self.agenda):
            score = itm.priority - (0.02 * ((time.time() - itm.created_at) / 10.0))
            if score > best_score:
                best_score = score
                best_idx = i
        return self.agenda.pop(best_idx) if best_idx is not None else None

    def _agenda_is_poor(self) -> bool:
        """Heuristique: pas d’items 'intake'/'learning'/'alignment' à priorité >= 0.6"""
        useful = [i for i in self.agenda if i.kind in ("intake", "learning", "alignment") and i.priority >= 0.6]
        if not useful:
            self.ticks_without_useful += 1
        else:
            self.ticks_without_useful = 0
        return self.ticks_without_useful >= self.FALLBACK_AFTER_TICKS

    def _inject_fallback_seed(self) -> None:
        for p in self.fallback_seed:
            self._push_if_new(p)
        self.ticks_without_useful = 0

    # ---------- Capteurs/état ----------

    def _detect_weak_capabilities(self) -> List[tuple]:
        """Retourne [(metric, score)] pour les métriques basses."""
        res: List[tuple] = []
        try:
            perf = (self.metacog.cognitive_monitoring.get("performance_tracking", {})
                    if self.metacog else {})
            # on lit la dernière valeur si dispo
            for metric, data in perf.items():
                if not data:
                    continue
                val = data[-1]["value"] if isinstance(data, list) and data else 0.0
                if val < 0.55:
                    res.append((metric, float(val)))
        except Exception:
            pass
        return res[:5]

    def _context_is_fuzzy(self) -> bool:
        """Vérifie s’il y a assez d’infos pour agir sans demander à l’utilisateur."""
        # Simple heuristique : pas de fichiers, pas de tâches intake >= 0.6, pas de user_msg récent (non accessible ici)
        has_intake = any(i for i in self.agenda if i.kind == "intake" and i.priority >= 0.6)
        return (not has_intake) and (not self._dir_has_content("./inbox"))

    def _maybe_emit_useful_question(self) -> None:
        questions = [i for i in self.agenda if i.kind == "alignment" and i.status == "queued"]
        if len(questions) >= self.MIN_USEFUL_QUESTIONS:
            return
        # Injecte une question courte et utile
        self._push_if_new({
            "title": "Question de clarification (priorités & contexte)",
            "kind": "alignment",
            "priority": 0.8,
            "rationale": "Réduire l’incertitude avant d’allouer des efforts.",
            "payload": {
                "action": "ask_user",
                "question": self._build_clarifying_question()
            }
        })

    # ---------- Actions concrètes ----------

    def _list_inbox(self, path: str) -> List[str]:
        try:
            return [f for f in os.listdir(path) if not f.startswith(".")]
        except Exception:
            return []

    def _snapshot_modules(self) -> Dict[str, Any]:
        snap = {"time": time.time(), "modules": {}, "constitution": self.constitution}
        for name in ("memory", "perception", "reasoning", "goals", "metacognition", "creativity", "world_model", "language"):
            obj = getattr(self.arch, name, None)
            snap["modules"][name] = {
                "present": obj is not None and not isinstance(obj, str),
                "attrs": sorted([a for a in dir(obj)])[:30] if obj else []
            }
        return snap

    def _propose_core_terms(self) -> List[str]:
        return [
            "objectif", "priorité", "rationale", "contexte",
            "contrainte", "hypothèse", "preuve", "feedback",
            "incertitude", "coût", "bénéfice", "itération"
        ]

    def _language_probe(self) -> Dict[str, Any]:
        report = {
            "can_parse": bool(self.language and hasattr(self.language, "parse_utterance")),
            "has_vocab": bool(self.language and hasattr(self.language, "known_terms")),
            "notes": []
        }
        if not report["can_parse"]:
            report["notes"].append("parse_utterance indisponible → clarifier l’API du module langage.")
        if not report["has_vocab"]:
            report["notes"].append("Pas de vocabulaire interne détecté → construire un glossaire initial.")
        return report

    # ---------- Helpers ----------

    def _build_clarifying_question(self) -> str:
        base = [
            "Quel est l’objectif le plus important pour toi maintenant ?",
            "Y a-t-il des contraintes (temps, format, sources) que je dois respecter ?",
            "Souhaites-tu que je priorise l’exploration ou la fiabilité ?"
        ]
        return " / ".join(base)

    def _dir_has_content(self, path: str) -> bool:
        try:
            return any(not f.startswith(".") for f in os.listdir(path))
        except Exception:
            return False

    def _write_json(self, path: str, data: Dict[str, Any]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"⚠️ Échec d’écriture JSON {path}: {e}")

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{stamp}] {msg}"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        # echo console minimal
        print(f"[Autonomy] {msg}")

"""Autonomy related helpers."""

from .core import AutonomyCore

__all__ = ["AutonomyCore"]
