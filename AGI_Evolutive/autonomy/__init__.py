# Gestion de l'autonomie : auto-seed d'objectifs, micro-constitution, agenda, déduplication et fallback
# Compatible avec l'architecture existante (GoalSystem, Metacognition, Memory, Perception, Language, etc.)
# Aucune dépendance externe (stdlib uniquement). Logs lisibles dans ./logs/autonomy.log

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
import os, time, uuid, json, threading, random, math

from AGI_Evolutive.utils.jsonsafe import json_sanitize

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

class AdaptiveEMA:
    """EMA avec choix dynamique de beta via Thompson Sampling."""

    def __init__(self,
                 betas: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
                 error_threshold: float = 0.05,
                 drift_threshold: float = 0.12,
                 forgetting: float = 0.975):
        self.betas = betas
        self.error_threshold = error_threshold
        self.drift_threshold = drift_threshold
        self.forgetting = forgetting
        self.state: Optional[float] = None
        self.last_smoothed: Optional[float] = None
        self.posteriors: Dict[float, Tuple[float, float]] = {
            b: [1.0, 1.0] for b in betas
        }

    def _sample_posterior(self, beta: float) -> float:
        a, b = self.posteriors[beta]
        return random.betavariate(max(a, 1e-3), max(b, 1e-3))

    def _decay_posteriors(self, beta: float) -> None:
        a, b = self.posteriors[beta]
        a = max(1.0, self.forgetting * a)
        b = max(1.0, self.forgetting * b)
        self.posteriors[beta] = [a, b]

    def update(self, value: float) -> Dict[str, float]:
        if value is None or math.isnan(value):
            return {
                "smoothed": self.state if self.state is not None else 0.0,
                "beta": self.betas[-1],
                "error": 0.0,
                "drift": 0.0,
                "error_threshold": self.error_threshold,
                "drift_threshold": self.drift_threshold,
            }

        if self.state is None:
            self.state = value
            self.last_smoothed = value
            return {
                "smoothed": value,
                "beta": self.betas[-1],
                "error": 0.0,
                "drift": 0.0,
                "error_threshold": self.error_threshold,
                "drift_threshold": self.drift_threshold,
            }

        sampled = {b: self._sample_posterior(b) for b in self.betas}
        chosen_beta = max(sampled.items(), key=lambda kv: kv[1])[0]
        self._decay_posteriors(chosen_beta)

        prev_state = self.state
        self.state = (chosen_beta * value) + ((1.0 - chosen_beta) * self.state)
        error = abs(self.state - value)
        drift = abs(self.state - (self.last_smoothed if self.last_smoothed is not None else prev_state))
        success = 1 if error <= self.error_threshold else 0
        a, b = self.posteriors[chosen_beta]
        self.posteriors[chosen_beta] = [a + success, b + (1 - success)]
        self.last_smoothed = self.state

        return {
            "smoothed": self.state,
            "beta": chosen_beta,
            "error": error,
            "drift": drift,
            "error_threshold": self.error_threshold,
            "drift_threshold": self.drift_threshold,
        }


class StreamingCorrelation:
    """Corrélation glissante avec facteur d'oubli."""

    def __init__(self, forgetting: float = 0.97):
        self.forgetting = forgetting
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.var_x = 1e-6
        self.var_y = 1e-6
        self.cov_xy = 0.0

    def update(self, x: float, y: float) -> float:
        decay = self.forgetting
        prev_mean_x = self.mean_x
        prev_mean_y = self.mean_y
        self.mean_x = (decay * self.mean_x) + ((1 - decay) * x)
        self.mean_y = (decay * self.mean_y) + ((1 - decay) * y)
        self.cov_xy = (decay * self.cov_xy) + ((1 - decay) * (x - prev_mean_x) * (y - prev_mean_y))
        self.var_x = (decay * self.var_x) + ((1 - decay) * (x - prev_mean_x) * (x - self.mean_x))
        self.var_y = (decay * self.var_y) + ((1 - decay) * (y - prev_mean_y) * (y - self.mean_y))

        if self.var_x <= 1e-8 or self.var_y <= 1e-8:
            return 0.0
        corr = max(-1.0, min(1.0, self.cov_xy / math.sqrt(self.var_x * self.var_y)))
        return corr


class OnlineWeightLearner:
    """Mise à jour en ligne type ridge pour pondérer les priorités."""

    def __init__(self, l2: float = 0.1, max_step: float = 0.05, forgetting: float = 0.98):
        self.l2 = l2
        self.max_step = max_step
        self.forgetting = forgetting
        self.weights: Dict[str, float] = defaultdict(lambda: 0.8)

    def update(self, key: str, feature: float, target: float) -> float:
        if feature <= 0:
            return self.weights[key]
        weight = self.weights[key] * self.forgetting
        prediction = weight * feature
        gradient = (target - prediction) * feature - (self.l2 * weight)
        step = max(-self.max_step, min(self.max_step, gradient))
        weight = max(0.1, min(2.5, weight + step))
        self.weights[key] = weight
        return weight


class MetricLearningState:
    """Suivi des métriques faibles avec lissage adaptatif et poids appris."""

    def __init__(self, name: str, forgetting: float = 0.97):
        self.name = name
        self.ema = AdaptiveEMA()
        self.last_raw: Optional[float] = None
        self.forgetting = forgetting
        self.correlation = StreamingCorrelation(forgetting=forgetting)
        self.last_corr = 0.0

    def observe(self, value: float, learner: OnlineWeightLearner) -> Dict[str, float]:
        ema_state = self.ema.update(value)
        improvement = 0.0 if self.last_raw is None else value - self.last_raw
        severity = max(0.0, 1.0 - ema_state["smoothed"])
        weight = learner.update(self.name, severity, max(0.0, -improvement))
        corr = self.correlation.update(ema_state["smoothed"], max(0.0, -improvement))
        self.last_raw = value

        return {
            "metric": self.name,
            "smoothed": ema_state["smoothed"],
            "beta": ema_state["beta"],
            "error": ema_state["error"],
            "drift": ema_state["drift"],
            "weight": weight,
            "correlation": corr,
            "severity": severity,
            "improvement": improvement,
            "error_threshold": ema_state.get("error_threshold", 0.05),
            "drift_threshold": ema_state.get("drift_threshold", 0.12),
        }


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
        self.FALLBACK_AFTER_TICKS: int = 8       # si rien d'utile émis → fallback
        self.MAX_QUEUE: int = 50
        self.MIN_USEFUL_QUESTIONS: int = 1       # toujours pousser un minimum de questions utiles
        self.LAST_N_DEDUPE: int = 40             # fenêtre de déduplication

        # Micro-constitution : principes (pas une todo-list)
        self.constitution: List[str] = [
            "Toujours expliciter ce qui manque (données, contraintes) avant d'agir.",
            "Optimiser le ratio progrès/coût (temps, confusion, dette).",
            "Améliorer en priorité les capacités générales (langage, raisonner, apprendre).",
            "Valider par boucles courtes: hypothèses → preuves/feedback.",
            "Respecter l'humain (clarté, coopération, sécurité)."
        ]

        # Fallback seed (au cas où l'auto-seed n'émet rien d'utile)
        self.fallback_seed: List[Dict[str, Any]] = [
            {
                "title": "Cartographier mes modules et leurs métriques",
                "kind": "meta",
                "priority": 0.9,
                "rationale": "Avoir une vue claire pour décider quoi améliorer en premier.",
                "payload": {"action": "snapshot_modules"}
            },
            {
                "title": "Analyser l'inbox et créer un plan d'intégration",
                "kind": "intake",
                "priority": 0.8,
                "rationale": "L'environnement est source de contexte et d'apprentissage.",
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
        self.metric_states: Dict[str, MetricLearningState] = {}
        self.weight_learner = OnlineWeightLearner()

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
        - Exécute (légèrement) certaines tâches "automatiques" (scan inbox, snapshot…)
        - Pousse les objectifs vers GoalSystem si présent
        """
        with self._lock:
            now = time.time()
            if now - self.last_tick < 0.5:
                return  # évite le spam si le cycle est très rapide
            self.last_tick = now

            # 1) Sème de nouveaux objectifs si l'agenda est pauvre
            self._maybe_seed()

            # 2) Évite la stagnation : s'il n'y a pas d'élément "utile", fallback
            if self._agenda_is_poor():
                self._log("⚠️ Agenda peu utile → fallback seed")
                self._inject_fallback_seed()

            # 3) Émet au moins une question utile si besoin
            self._maybe_emit_useful_question()

            # 4) Essaie de "démarrer" la prochaine tâche exécutable (automatique)
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
        for cap_state in weak:
            cap = cap_state["metric"]
            score = cap_state["smoothed"]
            props.append({
                "title": f'Améliorer la capacité "{cap}"',
                "kind": "learning",
                "priority": self._priority_from_metric(cap_state),
                "rationale": self._rationale_from_metric(cap_state),
                "payload": {"action": "improve_metric", "metric": cap}
            })

        # b) environnement (inbox)
        inbox_path = "./inbox"
        if os.path.isdir(inbox_path) and self._dir_has_content(inbox_path):
            props.append({
                "title": "Analyser l'inbox (fichiers récents)",
                "kind": "intake",
                "priority": 0.8,
                "rationale": "Nouveaux indices contextuels disponibles.",
                "payload": {"action": "scan_inbox", "path": inbox_path}
            })

        # c) langage / explication - toujours utile si pas de base lexicale
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
            # si module language inconnu → tâche d'investigation
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
                "rationale": "Constitution: expliciter ce qui manque avant d'agir.",
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
                # Ajoute sous-tâches d'intégration
                for name in listed[:20]:
                    self._push_if_new({
                        "title": f'Intégrer le fichier "{name}"',
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
                # rien d'autre à faire; la réponse utilisateur alimente la suite

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

    # ---------- Utilitaires d'agenda ----------

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
        # priorité simple (max priority, plus ancien en cas d'égalité)
        best_idx = None
        best_score = -1.0
        for i, itm in enumerate(self.agenda):
            score = itm.priority - (0.02 * ((time.time() - itm.created_at) / 10.0))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            return None
        best_item = self.agenda[best_idx]
        del self.agenda[best_idx]
        return best_item

    def _agenda_is_poor(self) -> bool:
        """Heuristique: pas d'items 'intake'/'learning'/'alignment' à priorité >= 0.6"""
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

    def _detect_weak_capabilities(self) -> List[Dict[str, float]]:
        """Retourne les métriques faibles avec lissage adaptatif et poids dynamiques."""
        res: List[Dict[str, float]] = []
        try:
            perf = (self.metacog.cognitive_monitoring.get("performance_tracking", {})
                    if self.metacog else {})
            # on lit la dernière valeur si dispo
            for metric, data in perf.items():
                if not data:
                    continue
                val = data[-1]["value"] if isinstance(data, list) and data else 0.0
                if metric not in self.metric_states:
                    self.metric_states[metric] = MetricLearningState(metric)
                state = self.metric_states[metric].observe(float(val), self.weight_learner)
                if state["smoothed"] < 0.7:
                    res.append(state)
                    self._maybe_log_metric_events(state)
        except Exception:
            pass
        res.sort(key=lambda s: (s["smoothed"], -s["weight"]))
        return res[:5]

    def _priority_from_metric(self, state: Dict[str, float]) -> float:
        base = 0.55 + (0.25 * state["weight"])
        severity = min(0.45, state["severity"] * 0.45)
        priority = min(0.98, base + severity)
        return priority

    def _rationale_from_metric(self, state: Dict[str, float]) -> str:
        metric = state["metric"]
        smoothed = state["smoothed"]
        beta = state["beta"]
        correlation = state["correlation"]
        return (
            f'La métrique "{metric}" est lissée à {smoothed:.2f} '
            f'(β adaptatif={beta:.2f}, corr={correlation:.2f}).'
        )

    def _maybe_log_metric_events(self, state: Dict[str, float]) -> None:
        drift = state.get("drift", 0.0)
        error = state.get("error", 0.0)
        corr = state.get("correlation", 0.0)
        metric = state["metric"]
        drift_threshold = state.get("drift_threshold", 0.12)
        error_threshold = state.get("error_threshold", 0.2)
        if drift > drift_threshold:
            self._log(f"📈 Drift détecté sur {metric} (Δ={drift:.3f})")
        if error > error_threshold:
            self._log(f"📉 Signal bruité pour {metric} (erreur={error:.3f})")
        prev_corr = getattr(self.metric_states[metric], "last_corr", 0.0)
        if abs(corr - prev_corr) > 0.05:
            trend = "↑" if corr > prev_corr else "↓"
            self._log(f"🔁 Corrélation {trend} pour {metric}: {prev_corr:.2f} → {corr:.2f}")
            self.metric_states[metric].last_corr = corr

    def _context_is_fuzzy(self) -> bool:
        """Vérifie s'il y a assez d'infos pour agir sans demander à l'utilisateur."""
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
            "rationale": "Réduire l'incertitude avant d'allouer des efforts.",
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
            report["notes"].append("parse_utterance indisponible → clarifier l'API du module langage.")
        if not report["has_vocab"]:
            report["notes"].append("Pas de vocabulaire interne détecté → construire un glossaire initial.")
        return report

    # ---------- Helpers ----------

    def _build_clarifying_question(self) -> str:
        base = [
            "Quel est l'objectif le plus important pour toi maintenant ?",
            "Y a-t-il des contraintes (temps, format, sources) que je dois respecter ?",
            "Souhaites-tu que je priorise l'exploration ou la fiabilité ?"
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
                json.dump(json_sanitize(data), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"⚠️ Échec d'écriture JSON {path}: {e}")

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
