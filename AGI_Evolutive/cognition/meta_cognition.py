import os, json, time
from typing import List, Dict, Any, Optional
from collections import Counter

class MetaCognition:
    """
    Évalue l'incertitude, repère des lacunes, génère des learning-goals,
    et enregistre des réflexions (inner-monologue).
    Persiste état dans data/metacog.json
    """
    def __init__(self, memory_store, planner, self_model, data_dir: str = "data"):
        self.memory = memory_store
        self.planner = planner
        self.self_model = self_model
        self.path = os.path.join(data_dir, "metacog.json")
        self.state = {
            "last_assessment_ts": 0.0,
            "open_questions": [],     # [{"q":"...", "topic":"...", "priority":0..1}]
            "uncertainty": 0.5,       # 0..1
            "domains": {},            # {"language": {"confidence":0.6,"gaps":["..."]}, ...}
            "history": []             # log compact
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

    # --------- Analyse & incertitude ---------
    def assess_understanding(self, horizon: int = 150) -> Dict[str, Any]:
        recents = self.memory.get_recent_memories(n=horizon)
        q_cnt = 0
        a_cnt = 0
        err_cnt = 0
        lessons = 0
        tokens = Counter()

        for m in recents:
            kind = (m.get("kind") or "").lower()
            text = (m.get("text") or "")
            if kind in ("question","interaction") and text.endswith("?"):
                q_cnt += 1
            if kind in ("lesson","reflection"):
                lessons += 1
            if "error" in kind:
                err_cnt += 1
            for w in text.lower().split():
                if w.isalpha(): tokens[w] += 1

            # heuristique de "réponse"/“affirmation apprise”
            if any(k in kind for k in ("lesson","insight")) or ("answer" in text.lower()):
                a_cnt += 1

        # incertitude : plus de questions et d’erreurs -> plus d’incertitude
        denom = max(1, q_cnt + a_cnt)
        uncertainty = 0.5 + 0.4*((q_cnt - a_cnt)/denom) + 0.1*min(1.0, err_cnt/10.0)
        uncertainty = max(0.0, min(1.0, uncertainty))

        # domaines rudimentaires à partir de tokens fréquents
        domains = {
            "language": {"confidence": max(0.2, 1.0 - uncertainty*0.6), "gaps": []},
            "humans":   {"confidence": max(0.1, 0.4 + 0.03*lessons - 0.02*q_cnt), "gaps": []},
            "planning": {"confidence": 0.5, "gaps": []}
        }

        # extraire thèmes/gaps probables
        common = [w for (w,_) in tokens.most_common(30)]
        gaps = []
        for w in common:
            # si mot fréquent mais peu de leçons → probablement superficiel
            if lessons < 3 and w not in ("et","le","la","de","des","les","un","une","du","en","à"):
                gaps.append(w)
        for k in domains:
            domains[k]["gaps"] = gaps[:5]

        out = {
            "uncertainty": float(uncertainty),
            "domains": domains,
            "stats": {"questions": q_cnt, "answers": a_cnt, "errors": err_cnt, "lessons": lessons}
        }
        self.state["uncertainty"] = out["uncertainty"]
        self.state["domains"] = domains
        self.state["last_assessment_ts"] = time.time()
        self.state["history"].append({"ts": self.state["last_assessment_ts"], "uncertainty": self.state["uncertainty"]})
        self._save()
        return out

    # --------- Génération de learning goals ---------
    def propose_learning_goals(self, max_goals: int = 3) -> List[Dict[str, Any]]:
        assessment = self.assess_understanding()
        uncertainty = assessment["uncertainty"]
        candidate_goals = []
        for domain, d in assessment["domains"].items():
            for g in d.get("gaps", []):
                prio = 0.5*uncertainty + 0.5*(1.0 - d["confidence"])
                candidate_goals.append(
                    {"id": f"learn_{domain}_{g}", "desc": f"Comprendre le concept '{g}' dans le domaine {domain}", "priority": prio}
                )
        candidate_goals.sort(key=lambda x: x["priority"], reverse=True)
        goals = candidate_goals[:max_goals]

        # Planifier des micro-actions pour chaque goal
        for goal in goals:
            plan = self.planner.plan_for_goal(goal["id"], goal["desc"])
            if not plan["steps"]:
                self.planner.add_step(goal["id"], f"Collecter exemples concrets de '{goal['desc']}'")
                self.planner.add_step(goal["id"], f"Poser 2 questions ciblées sur '{goal['desc']}'")
        # journaliser en mémoire
        if goals:
            self.memory.add_memory({
                "kind": "reflection",
                "text": f"Génération de {len(goals)} learning-goals basés sur incertitude",
                "goals": goals,
                "ts": time.time()
            })
        return goals

    # --------- Journal réflexif ---------
    def log_inner_monologue(self, text: str, tags: Optional[List[str]] = None):
        self.memory.add_memory({
            "kind": "reflection",
            "text": text,
            "tags": tags or [],
            "ts": time.time()
        })
