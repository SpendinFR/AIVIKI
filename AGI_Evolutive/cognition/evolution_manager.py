import json
import os
import statistics
import threading
import time
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _safe_write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_read_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _mean(xs: List[float], default: float = 0.0) -> float:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(statistics.fmean(xs)) if xs else float(default)


def _rolling(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    tail = values[-k:] if len(values) > k else values
    return _mean(tail)


class EvolutionManager:
    """
    Gestion long-terme de l'agent :
      - collecte des métriques cross-modules
      - détection de tendances (amélioration / régression / stagnation)
      - analyse des risques (fatigue, charge, erreurs récurrentes)
      - recommandations stratégiques (réglages, priorités, expérimentations)
      - propositions d'évolution (proposals) -> à valider par une policy si elle existe
    Persistance :
      - data/evolution/state.json
      - data/evolution/cycles.jsonl   (trace de chaque cycle)
      - data/evolution/recommendations.jsonl
      - data/evolution/dashboard.json (snapshot consolidé)
    """

    def __init__(self, data_dir: str = "data", horizon_cycles: int = 200):
        self.data_dir = data_dir
        self.paths = {
            "state": os.path.join(self.data_dir, "evolution/state.json"),
            "cycles": os.path.join(self.data_dir, "evolution/cycles.jsonl"),
            "reco": os.path.join(self.data_dir, "evolution/recommendations.jsonl"),
            "dashboard": os.path.join(self.data_dir, "evolution/dashboard.json"),
        }
        os.makedirs(os.path.dirname(self.paths["state"]), exist_ok=True)

        # liens (optionnels) vers l'archi
        self.arch = None
        self.memory = None
        self.metacog = None
        self.goals = None
        self.learning = None
        self.emotions = None
        self.language = None

        # État persistant
        self.state = _safe_read_json(self.paths["state"], {
            "created_at": _now(),
            "last_cycle_id": 0,
            "history": {
                # séries brutes
                "reasoning_speed": [],
                "reasoning_confidence": [],
                "learning_rate": [],
                "recall_accuracy": [],
                "cognitive_load": [],
                "fatigue": [],
                "error_rate": [],
                "goals_progress": [],
                "affect_valence": [],
            },
            "milestones": [],  # liste d'événements marquants
            "risk_flags": [],  # ex: "regression_learning", "high_fatigue"
        })
        self.horizon = horizon_cycles
        self._state_lock = threading.RLock()

    # ---------- binding ----------
    def bind(self, architecture=None, memory=None, metacog=None,
             goals=None, learning=None, emotions=None, language=None):
        self.arch = architecture
        self.memory = memory
        self.metacog = metacog
        self.goals = goals
        self.learning = learning
        self.emotions = emotions
        self.language = language

    # ---------- cycle ingestion ----------
    def record_cycle(self, extra_tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Capture l'état courant des métriques utiles, loggue en JSONL, met à jour l'état.
        Appelle ensuite evaluate_cycle() pour calculer tendances + risques.
        """
        with self._state_lock:
            cycle_id = self.state["last_cycle_id"] + 1
            snap = {
                "t": _now(),
                "cycle_id": cycle_id,
                "metrics": self._collect_metrics_snapshot(),
                "tags": extra_tags or {}
            }
            _append_jsonl(self.paths["cycles"], snap)

            # pousser les séries
            hist = self.state["history"]
            m = snap["metrics"]
            hist["reasoning_speed"].append(m.get("reasoning_speed", 0.0))
            hist["reasoning_confidence"].append(m.get("reasoning_confidence", 0.0))
            hist["learning_rate"].append(m.get("learning_rate", 0.0))
            hist["recall_accuracy"].append(m.get("recall_accuracy", 0.0))
            hist["cognitive_load"].append(m.get("cognitive_load", 0.0))
            hist["fatigue"].append(m.get("fatigue", 0.0))
            hist["error_rate"].append(m.get("error_rate", 0.0))
            hist["goals_progress"].append(m.get("goals_progress", 0.0))
            hist["affect_valence"].append(m.get("affect_valence", 0.0))

            # limiter l'horizon
            for k in hist.keys():
                if len(hist[k]) > self.horizon:
                    hist[k] = hist[k][-self.horizon:]

            self.state["last_cycle_id"] = cycle_id

            # évaluation et recommandations
            eval_out = self.evaluate_cycle()
            dash = self._make_dashboard_snapshot()
            _safe_write_json(self.paths["dashboard"], dash)
            _safe_write_json(self.paths["state"], self.state)
            return {"snapshot": snap, "evaluation": eval_out, "dashboard": dash}

    # ---------- collect ----------
    def _collect_metrics_snapshot(self) -> Dict[str, float]:
        """
        Aggrège les métriques cross-modules avec garde-fous.
        """
        out = {
            "reasoning_speed": 0.5,
            "reasoning_confidence": 0.5,
            "learning_rate": 0.3,
            "recall_accuracy": 0.5,
            "cognitive_load": 0.5,
            "fatigue": 0.2,
            "error_rate": 0.0,
            "goals_progress": 0.0,
            "affect_valence": 0.0,
        }

        # métacognition → performance_tracking
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            perf = self.metacog.cognitive_monitoring.get("performance_tracking", {})
            def last(metric, default=0.5):
                arr = perf.get(metric, [])
                if not arr:
                    return default
                try:
                    return float(arr[-1]["value"])
                except Exception:
                    return default
            out["reasoning_speed"] = last("reasoning_speed", 0.5)
            out["reasoning_confidence"] = last("reasoning_confidence", 0.5)
            out["learning_rate"] = last("learning_rate", 0.3)
            out["recall_accuracy"] = last("recall_accuracy", 0.5)

        # ressources cognitives (via moniteur)
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            try:
                rm = self.metacog.cognitive_monitoring.get("resource_monitoring")
                if rm:
                    # ré-évaluer à la volée si possible
                    load = rm.assess_cognitive_load(self.arch, getattr(self.arch, "reasoning", None))
                    out["cognitive_load"] = float(load)
            except Exception:
                pass

        # fatigue → metacog resource_monitoring
        if self.metacog and hasattr(self.metacog, "cognitive_monitoring"):
            try:
                rm = self.metacog.cognitive_monitoring.get("resource_monitoring")
                if rm:
                    fat = rm.assess_fatigue(self.metacog.metacognitive_history, self.arch)
                    out["fatigue"] = float(fat)
            except Exception:
                pass

        # erreur rate (comptage récent)
        if self.metacog:
            try:
                corr = list(self.metacog.metacognitive_history.get("error_corrections", []))
                recent = corr[-50:] if len(corr) > 50 else corr
                out["error_rate"] = len(recent) / max(10.0, (recent[-1]["timestamp"] - recent[0]["timestamp"])) if len(recent) > 1 else 0.0
            except Exception:
                pass

        # progression des buts (si goal system expose une métrique)
        if self.goals and hasattr(self.goals, "get_progress"):
            try:
                out["goals_progress"] = float(self.goals.get_progress() or 0.0)
            except Exception:
                pass

        # valence émotionnelle globale (si dispo)
        if self.emotions and hasattr(self.emotions, "get_affect"):
            try:
                aff = self.emotions.get_affect()  # ex: dict {"valence": -1..1, "arousal": 0..1}
                if isinstance(aff, dict):
                    out["affect_valence"] = max(0.0, min(1.0, (float(aff.get("valence", 0.0)) + 1.0) / 2.0))
            except Exception:
                pass

        return out

    # ---------- evaluation ----------
    def evaluate_cycle(self) -> Dict[str, Any]:
        """
        Analyse tendances courtes et longues; pose des flags de risque; émet des recommandations.
        """
        with self._state_lock:
            hist = self.state["history"]
            eval_out = {
                "t": _now(),
                "rolling": {
                    "speed_5": _rolling(hist["reasoning_speed"], 5),
                    "learn_5": _rolling(hist["learning_rate"], 5),
                    "conf_5": _rolling(hist["reasoning_confidence"], 5),
                    "recall_5": _rolling(hist["recall_accuracy"], 5),
                    "fatigue_5": _rolling(hist["fatigue"], 5),
                    "load_5": _rolling(hist["cognitive_load"], 5),
                    "errors_20": _rolling(hist["error_rate"], 20),
                },
                "flags": [],
                "recommendations": []
            }

            r = eval_out["rolling"]
            # flags
            if r["learn_5"] < 0.35 and r["conf_5"] < 0.45:
                eval_out["flags"].append("learning_regression")
            if r["fatigue_5"] > 0.7:
                eval_out["flags"].append("high_fatigue")
            if r["load_5"] > 0.8:
                eval_out["flags"].append("overload")
            if r["errors_20"] > 0.05:
                eval_out["flags"].append("high_error_rate")

            # recos : stratégie & expérimentation
            eval_out["recommendations"] += self._strategy_recommendations(r)
            eval_out["recommendations"] += self._exploration_recommendations()

            # persiste recos + flags en JSONL
            _append_jsonl(self.paths["reco"], {
                "t": eval_out["t"],
                "cycle_id": self.state["last_cycle_id"],
                "flags": eval_out["flags"],
                "recommendations": eval_out["recommendations"]
            })

            # ajoute aux risk_flags si nouveau
            for f in eval_out["flags"]:
                if f not in self.state["risk_flags"]:
                    self.state["risk_flags"].append(f)
            return eval_out

    def _strategy_recommendations(self, r: Dict[str, float]) -> List[Dict[str, Any]]:
        recos = []
        # calibrage effort/attention
        if r["load_5"] > 0.8 or r["fatigue_5"] > 0.7:
            recos.append({
                "kind": "effort_downshift",
                "reason": "High cognitive load/fatigue",
                "action": "Réduire la profondeur des tâches; augmenter micro-pauses; privilégier consolidation."
            })
        # booster apprentissage si stagnation
        if r["learn_5"] < 0.4 and r["conf_5"] < 0.5:
            recos.append({
                "kind": "strategy_change",
                "reason": "Low learning rate and low confidence",
                "action": "Tester stratégie d'étude alternative (auto-explication, exemples concrets, retrieval practice)."
            })
        # renforcer rappel si recall bas
        if r["recall_5"] < 0.5:
            recos.append({
                "kind": "memory_consolidation",
                "reason": "Low recall accuracy",
                "action": "Augmenter fréquence de consolidation et répétition espacée."
            })
        # erreurs hautes
        if r["errors_20"] > 0.05:
            recos.append({
                "kind": "error_clinic",
                "reason": "High recent error rate",
                "action": "Lancer mini-clinique d'erreurs: catégoriser 10 dernières erreurs et élaborer correctifs ciblés."
            })
        return recos

    def _exploration_recommendations(self) -> List[Dict[str, Any]]:
        """
        S'appuie (si dispo) sur concept_graph / episodes pour suggérer des axes d'exploration.
        """
        recos = []
        # lire concept_graph si présent
        cgraph_path = os.path.join(self.data_dir, "concept_graph.json")
        cgraph = _safe_read_json(cgraph_path, {"nodes": {}, "edges": {}})
        nodes = cgraph.get("nodes", {})
        if nodes:
            # sélectionner concepts fréquents mais peu exploités (heuristique: counts faibles dans plans?)
            top_concepts = sorted(nodes.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:10]
            for name, meta in top_concepts[:3]:
                recos.append({
                    "kind": "explore_concept",
                    "concept": name,
                    "reason": "Concept central du vécu récent",
                    "action": f"Créer un sous-objectif d'analyse approfondie du concept '{name}'."
                })
        # lire derniers épisodes pour détecter chaînes cause→effet
        episodes_path = os.path.join(self.data_dir, "episodes.jsonl")
        if os.path.exists(episodes_path):
            try:
                last_lines = []
                with open(episodes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        last_lines.append(json.loads(line))
                last_lines = last_lines[-5:] if len(last_lines) > 5 else last_lines
                if last_lines:
                    recos.append({
                        "kind": "reflect_episode",
                        "reason": "Épisodes récents disponibles",
                        "action": "Déclencher réflexion ciblée sur 1-2 épisodes pour extraire leçons causales."
                    })
            except Exception:
                pass
        return recos

    # ---------- proposals (optionnels) ----------
    def propose_evolution(self) -> List[Dict[str, Any]]:
        """
        Génère des 'proposals' d'évolution (self_model, goals, stratégies).
        Ces objets sont **à valider** par une policy/contrôle en amont si dispo.
        Retourne la liste des proposals émis.
        """
        proposals = []

        # Exemple 1 : ajuster poids de curiosité si learning_rate bas
        lr = _rolling(self.state["history"]["learning_rate"], 8)
        if lr < 0.4:
            proposals.append({
                "type": "adjust_drive",
                "target": "curiosity",
                "delta": +0.1,
                "rationale": "Learning rate bas sur 8 cycles - stimuler exploration."
            })

        # Exemple 2 : créer macro-goal d'étude d'un concept central
        cgraph_path = os.path.join(self.data_dir, "concept_graph.json")
        cgraph = _safe_read_json(cgraph_path, {"nodes": {}, "edges": {}})
        if cgraph.get("nodes"):
            top = sorted(cgraph["nodes"].items(), key=lambda kv: kv[1].get("count", 0), reverse=True)[:1]
            if top:
                concept = top[0][0]
                proposals.append({
                    "type": "create_goal",
                    "title": f"Étudier en profondeur : {concept}",
                    "criteria": ["résumer en 5 points", "exemples concrets", "quiz de rappel >= 0.7"],
                    "value": 0.6,
                    "rationale": "Concept central du vécu récent"
                })

        # Exemple 3 : calibration métacognitive si écart confiance/perf
        conf = _rolling(self.state["history"]["reasoning_confidence"], 8)
        recall = _rolling(self.state["history"]["recall_accuracy"], 8)
        if conf - recall > 0.25:
            proposals.append({
                "type": "metacog_calibration",
                "action": "Ajouter exercice de calibration confiance↔rappel (prédire score avant test).",
                "rationale": "Surconfiance détectée (conf - recall > 0.25)"
            })

        # Publier en mémoire (si dispo) pour traçabilité + éventuelle policy
        if self.memory and hasattr(self.memory, "add_memory"):
            for p in proposals:
                try:
                    self.memory.add_memory(
                        kind="evolution_proposal",
                        content=p.get("title") or p.get("type"),
                        metadata={"proposal": p, "source": "EvolutionManager", "timestamp": _now()},
                    )
                except Exception:
                    pass
        return proposals

    # ---------- dashboard ----------
    def _make_dashboard_snapshot(self) -> Dict[str, Any]:
        h = self.state["history"]
        snap = {
            "t": _now(),
            "last_cycle": self.state["last_cycle_id"],
            "rolling": {
                "speed_8": _rolling(h["reasoning_speed"], 8),
                "learn_8": _rolling(h["learning_rate"], 8),
                "conf_8": _rolling(h["reasoning_confidence"], 8),
                "recall_8": _rolling(h["recall_accuracy"], 8),
                "load_8": _rolling(h["cognitive_load"], 8),
                "fatigue_8": _rolling(h["fatigue"], 8),
                "goals_8": _rolling(h["goals_progress"], 8),
            },
            "risk_flags": list(self.state.get("risk_flags", [])),
            "milestones": list(self.state.get("milestones", []))[-20:]
        }
        return snap

    # ---------- public helpers ----------
    def get_long_term_trends(self) -> Dict[str, float]:
        return self._make_dashboard_snapshot().get("rolling", {})

    def export_dashboard(self) -> Dict[str, Any]:
        dash = self._make_dashboard_snapshot()
        _safe_write_json(self.paths["dashboard"], dash)
        return dash
from typing import Dict, Any, List

class EvolutionManager:
    """
    Suit les performances de cycle en cycle et propose des ajustements macro.
    Persiste dans data/evolution.json
    """
    def __init__(self, data_path: str = "data/evolution.json"):
        self.path = data_path
        self.state = {
            "cycle_count": 0,
            "metrics_history": [],   # [{"ts": <timestamp>, "intr": <float>, "extr": <float>, "learn": <float>, "uncert": <float>}]
            "strategies": []         # notes d'ajustement
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def log_cycle(self, intrinsic: float, extrinsic: float, learning_rate: float, uncertainty: float):
        self.state["cycle_count"] += 1
        self.state["metrics_history"].append({
            "ts": time.time(),
            "intr": float(intrinsic),
            "extr": float(extrinsic),
            "learn": float(learning_rate),
            "uncert": float(uncertainty)
        })
        self.state["metrics_history"] = self.state["metrics_history"][-500:]
        self._save()

    def propose_macro_adjustments(self) -> List[str]:
        mh = self.state["metrics_history"]
        if len(mh) < 10: return []
        last = mh[-10:]
        avg_unc = statistics.fmean(x["uncert"] for x in last)
        avg_learn = statistics.fmean(x["learn"] for x in last)
        notes: List[str] = []
        if avg_unc > 0.65:
            notes.append("Augmenter exploration (curiosity), planifier plus de questions ciblées.")
        if avg_learn < 0.45:
            notes.append("Changer stratégie d'étude: plus d'exemples concrets et feedback.")
        if not notes:
            notes.append("Maintenir les stratégies actuelles, progression stable.")
        self.state["strategies"].append({"ts": time.time(), "notes": notes})
        self._save()
        return notes
