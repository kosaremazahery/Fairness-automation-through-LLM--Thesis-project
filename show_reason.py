"""
show_reasons.py — print what the model flagged, and WHY, for the latest runs.

    python show_reasons.py          # latest 2 successful runs
    python show_reasons.py 4        # latest 4

Each line: [Charter label | GDPR label] column — the model's reason.
SENS = sensitive in that answer key, no = non-sensitive, ??? = not a column (invented).

Note: the model only gives reasons for what it FLAGGED. It never explains what it left out,
so a missed column (e.g. race under the GDPR prompt) has no reason to show. To see the
model's full answer, open llm_calls.json in the run's artifacts (mlflow ui).
"""

import json
import os
import sys

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(
    experiment_names=["llm-fairness-warnings"],
    filter_string="tags.status = 'ok'",
    order_by=["start_time DESC"],
    max_results=N,
)
if runs.empty:
    sys.exit("No successful runs found. Run python run_sweep.py first, from this folder.")

keys = {}
if os.path.exists(os.path.join("gold", "compas.json")):
    with open(os.path.join("gold", "compas.json"), encoding="utf-8") as f:
        keys = {c["feature"]: c for c in json.load(f)["activity_1"]["columns"]}


def label(feature):
    c = keys.get(feature)
    if c is None:
        return "  ???  |  ???  "
    return f"{'SENS ' if c['charter'] else ' no  '}  | {'SENS ' if c['gdpr'] else ' no  '} "


def fmt(x):
    return f"{x:.2f}" if x == x else "-"


for _, run in runs.iterrows():
    print("=" * 90)
    scores = []
    for b in ("charter", "gdpr"):
        if f"metrics.{b}_f1" in run and run[f"metrics.{b}_f1"] == run[f"metrics.{b}_f1"]:
            scores.append(f"{b}: P {fmt(run[f'metrics.{b}_precision'])} R {fmt(run[f'metrics.{b}_recall'])} "
                          f"F1 {fmt(run[f'metrics.{b}_f1'])}")
    temp = run.get("params.temperature", "?")
    seed = run.get("params.seed", "?")
    print(f"{run['params.model']} | prompt: {run['params.prompt_variant']} | temp {temp} seed {seed} | "
          + " | ".join(scores))
    print("=" * 90)
    warnings = mlflow.artifacts.load_dict(f"runs:/{run['run_id']}/warnings.json")
    if not warnings:
        print("(the model flagged NO columns, so there are no reasons to show)")
    else:
        print("[Charter| GDPR ]")
        for w in warnings:
            ground = f" <{w['ground']}>" if w.get("ground") else ""
            print(f"[{label(w['feature'])}] {w['feature']:<24}{ground} {w['message']}")
    print()