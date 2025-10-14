import os
import json
import argparse
import time
from datetime import datetime


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bar(x: float, width: int = 20) -> str:
    x = max(0.0, min(1.0, x))
    n = int(x * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {x:.2f}"


def section(title: str):
    print("\n" + title)
    print("-" * len(title))


def console_once():
    reasoning = read_jsonl("logs/reasoning.jsonl")
    experiments = read_jsonl("logs/experiments.jsonl")
    dag = read_json("logs/goals_dag.json")
    metacog = read_jsonl("logs/metacog.log")

    section("⏱ Statut")
    print("Maintenant:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    section("🧠 Raisonnement (derniers 5)")
    for r in reasoning[-5:]:
        goal = r.get("goal") or {}
        sol = r.get("solution")
        conf = r.get("final_confidence", 0.5)
        t = r.get("reasoning_time", 0.0)
        print(
            f"- but={goal.get('title', '?')} | conf={conf:.2f} | t={t:.2f}s | "
            f"solution={str(sol)[:80]}"
        )

    section("🎯 Objectifs (top 5 actifs)")
    if dag:
        nodes = dag.get("nodes", {})
        active = [
            n
            for n in nodes.values()
            if n.get("status") == "active" and n.get("progress", 0) < 1
        ]
        active.sort(
            key=lambda x: (x.get("value", 0), 1 - x.get("progress", 0)), reverse=True
        )
        for n in active[:5]:
            print(f"- {n['goal_id']}: {n['description']}")
            print(
                f"  progress {bar(n.get('progress', 0))} | "
                f"competence {bar(n.get('competence', 0))} | "
                f"value {bar(n.get('value', 0))}"
            )

    section("🧪 Expériences (derniers 3)")
    for e in experiments[-10:]:
        if "outcome" in e:
            o = e["outcome"]
            print(
                f"- Résultat {o['metric']} : {'OK' if o['success'] else 'KO'} "
                f"({o['observed']:.2f} vs {o['goal']:.2f})"
            )
        else:
            print(
                f"- Plan: {e.get('metric', '?')} baseline={e.get('baseline', '?')} "
                f"target={e.get('target_change', '?')} plan={e.get('plan', {})}"
            )

    section("📈 Métacog (derniers 5 événements)")
    for m in metacog[-5:]:
        ts = m.get("timestamp", time.time())
        print(
            f"- {datetime.fromtimestamp(ts).strftime('%H:%M:%S')} "
            f"{m.get('event_type', '?')}: {m.get('description', '')[:80]}"
        )


def export_html(path="logs/dashboard.html"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reasoning = read_jsonl("logs/reasoning.jsonl")
    dag = read_json("logs/goals_dag.json")

    rows = []
    for r in reasoning[-20:]:
        rows.append(
            f"<tr><td>{r.get('goal', {}).get('title', '')}</td>"
            f"<td>{r.get('final_confidence', 0.0):.2f}</td>"
            f"<td>{r.get('reasoning_time', 0.0):.2f}s</td>"
            f"<td><code>{str(r.get('solution'))[:80]}</code></td></tr>"
        )

    goals_rows = []
    if dag:
        nodes = dag.get("nodes", {})
        active = [
            n
            for n in nodes.values()
            if n.get("status") == "active" and n.get("progress", 0) < 1
        ]
        active.sort(
            key=lambda x: (x.get("value", 0), 1 - x.get("progress", 0)), reverse=True
        )
        for n in active[:10]:
            goals_rows.append(
                f"<tr><td>{n['goal_id']}</td><td>{n['description']}</td>"
                f"<td>{n.get('progress', 0):.2f}</td>"
                f"<td>{n.get('competence', 0):.2f}</td>"
                f"<td>{n.get('value', 0):.2f}</td></tr>"
            )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>AGI Dashboard</title>
<style>
body{{font-family: system-ui, sans-serif; margin:20px}}
h2{{margin-top:28px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px;font-size:14px}}
th{{background:#fafafa;text-align:left}}
code{{white-space:pre-wrap}}
</style></head>
<body>
<h1>AGI Dashboard</h1>
<p>Généré: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>Raisonnements récents</h2>
<table><thead><tr><th>But</th><th>Confiance</th><th>Durée</th><th>Solution</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>

<h2>Objectifs actifs</h2>
<table><thead><tr><th>ID</th><th>Description</th><th>Progress</th><th>Competence</th><th>Value</th></tr></thead>
<tbody>
{''.join(goals_rows)}
</tbody></table>

</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML exporté → {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Affiche un snapshot console et quitte")
    ap.add_argument("--html", action="store_true", help="Exporte un HTML statique dans logs/dashboard.html")
    args = ap.parse_args()

    if args.once or (not args.once and not args.html):
        console_once()
    if args.html:
        export_html()
