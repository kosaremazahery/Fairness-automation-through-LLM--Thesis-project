"""
check_setup.py — run this BEFORE run_sweep.py, from inside the project folder.

    python check_setup.py

Each check prints PASS or FAIL with what to do. Nothing is logged to MLflow.
"""

import json
import os
import sys

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

MODEL = "llama3.2:3b"
CSV = "compas-scores-two-years_v1.csv"
ADULT_CSV = "adult.csv"          # UCI Adult, train + test combined (Kaggle/OpenML copy)
results = []


def check(name, ok, fix=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or not fix else f"\n       -> {fix}"))
    return ok


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


print(f"Folder: {os.getcwd()}\n")

# 1. Right folder, all files present
files = ["run_sweep.py", "activity1.py", "scoring.py", "rescore.py", "llm_fairness.py", "probe_prompt.py",
         "proxy.py", CSV, os.path.join("gold", "compas.json"),
         os.path.join("gold", "compas_proxies_literature.json"),
         ADULT_CSV, os.path.join("gold", "adult.json")]
missing = [f for f in files if not os.path.exists(f)]
check("All project files are in this folder", not missing,
      f"missing: {missing}. cd into llm-fairness-compas, or move these files here.")

# 2. New versions, not old ones
if os.path.exists("activity1.py"):
    check("activity1.py is the new version (ground field)", "GROUNDS_FOR" in read("activity1.py"),
          "this is an OLD activity1.py. Replace it with the new one.")
if os.path.exists("activity1.py"):
    check("activity1.py has the 3000-token answer limit", "MAX_TOKENS = 3000" in read("activity1.py"),
          "this activity1.py still has the 1500-token limit. Replace it with the new one.")
if os.path.exists("scoring.py"):
    check("scoring.py is the new version (parse failures, definition echo)",
          "definition_echo" in read("scoring.py"),
          "this is an OLD scoring.py. Replace it with the new one.")
if os.path.exists("probe_prompt.py"):
    check("probe_prompt.py is the new version", "MAX_TOKENS" in read("probe_prompt.py"),
          "this is an OLD probe_prompt.py. Replace it with the new one.")
if os.path.exists("run_sweep.py"):
    check("run_sweep.py is the new version (temperature + seeds)", "SAMPLING" in read("run_sweep.py"),
          "this is an OLD run_sweep.py. Replace it with the new one.")
if os.path.exists("rescore.py"):
    check("rescore.py is the new version (groups by temperature)", '"temperature"' in read("rescore.py"),
          "this is an OLD rescore.py. Replace it with the new one.")
if os.path.exists("llm_fairness.py"):
    check("llm_fairness.py is the new version", "num_ctx" in read("llm_fairness.py"),
          "this is the OLD llm_fairness.py. Replace it with the new one.")

# 3. Python packages
try:
    import pandas as pd
    import mlflow  # noqa: F401
    import ollama
    check("Packages pandas, mlflow, ollama import", True)
except ImportError as e:
    check("Packages pandas, mlflow, ollama import", False,
          f"{e}. Activate the venv (.\\.venv\\Scripts\\Activate.ps1), then pip install pandas mlflow ollama")
    sys.exit(1)
try:
    import sklearn  # noqa: F401
    check("Package scikit-learn imports (needed by proxy.py)", True)
except ImportError:
    check("Package scikit-learn imports (needed by proxy.py)", False, "pip install scikit-learn")

# 4. Data and gold set
if not missing:
    df = pd.read_csv(CSV).dropna(thresh=2)
    check(f"CSV loads with 7214 rows (got {len(df)})", len(df) == 7214, "unexpected row count; check the CSV.")
    gold = json.loads(read(os.path.join("gold", "compas.json")))
    a1 = gold["activity_1"]
    names = {c["feature"] for c in a1["columns"]}
    check("Gold set is the binary format", "bases" in a1,
          "gold/compas.json is the OLD tiered version. Replace it with the new one.")
    check(f"Gold set has grounds and families (version {gold.get('version')})",
          "families" in a1 and all("charter_grounds" in c for c in a1["columns"] if c["charter"]),
          "gold/compas.json is older than 1.1-draft. Replace it with the new one.")
    check("Gold set covers every CSV column", names == set(df.columns),
          f"not in gold: {sorted(set(df.columns) - names)}; not in CSV: {sorted(names - set(df.columns))}")

    # 4b. Adult: same checks as COMPAS
    adf = pd.read_csv(ADULT_CSV).dropna(thresh=2)
    check(f"Adult CSV loads with 48842 rows (got {len(adf)})", len(adf) == 48842,
          "unexpected row count; check adult.csv (train + test combined, header row).")
    agold = json.loads(read(os.path.join("gold", "adult.json")))
    aa1 = agold["activity_1"]
    anames = {c["feature"] for c in aa1["columns"]}
    check(f"Adult gold set has the COMPAS structure (version {agold.get('version')})",
          list(agold) == list(gold) and list(aa1) == list(a1)
          and all("charter_grounds" in c for c in aa1["columns"] if c["charter"]),
          "gold/adult.json does not follow the gold/compas.json structure. Replace it with the new one.")
    check("Adult gold set covers every CSV column", anames == set(adf.columns),
          f"not in gold: {sorted(set(adf.columns) - anames)}; not in CSV: {sorted(anames - set(adf.columns))}")

    # 5. Parser and scoring on fake answers (no model needed)
    from activity1 import parse_warnings
    from scoring import score_warnings
    cols = list(df.columns)
    w, ok = parse_warnings('{"sensitive_features":[{"feature":"Race","reason":"r"},'
                           '{"feature":"sex","reason":"s"},{"feature":"religion","reason":"x"}]}', cols)
    check("Parser reads a JSON answer and fixes capitalisation",
          ok and [x["feature"] for x in w] == ["race", "sex", "religion"])
    _, ok_bad = parse_warnings("1. race\n2. sex", cols)
    check("Parser flags a non-JSON answer as parse_ok=False (no crash)", ok_bad is False)
    m, _ = score_warnings(w, gold, "charter")
    g, _ = score_warnings(w, gold, "gdpr")
    check(f"Lenient scoring works (Charter recall {m['recall']:.2f}, GDPR false alarms {g['fp']}, "
          f"hallucinated {m['hallucinated']})",
          m["tp"] == 2 and m["hallucinated"] == 1 and g["tp"] == 1 and g["fp"] == 1
          and "strict_f1" not in m)

    # strict scoring + error typing: dob with the wrong ground, priors flagged under GDPR
    w2, _ = parse_warnings(json.dumps({"sensitive_features": [
        {"feature": "race", "ground": "race", "reason": "r"},
        {"feature": "dob", "ground": "birth", "reason": "r"},
        {"feature": "priors_count", "ground": "none of the listed grounds", "reason": "r"},
    ]}), cols)
    s, d = score_warnings(w2, gold, "charter")
    check(f"Strict scoring works (lenient tp {s['tp']}, strict tp {s['strict_tp']}, "
          f"wrong ground {d['wrong_ground']}, off-definition {s['off_definition']})",
          s["tp"] == 2 and s["strict_tp"] == 1 and d["wrong_ground"] == {"dob": "birth"}
          and s["off_definition"] == 1)
    w3, _ = parse_warnings(json.dumps({"sensitive_features": [
        {"feature": "race", "ground": "racial or ethnic origin", "reason": "r"},
        {"feature": "priors_count", "ground": "data concerning health", "reason": "r"},
        {"feature": "c_charge_desc", "ground": "data concerning health", "reason": "r"},
    ]}), cols)
    t, _ = score_warnings(w3, gold, "gdpr")
    check(f"False alarms typed by family (criminal_history {t['fp_criminal_history']}, "
          f"current_charge {t['fp_current_charge']}, art10 {t['fp_art10']})",
          t["strict_tp"] == 1 and t["fp_art10"] == 2 and t["fp_criminal_history"] == 1)

    # empty answer to a legal prompt: precision undefined, recall/F1 zero, still strict-scored
    e, _ = score_warnings([], gold, "gdpr", grounded=True)
    check(f"Empty answer scored as undefined precision (P {e['precision']}, R {e['recall']}, "
          f"F1 {e['f1']}, strict R {e['strict_recall']}, empty {e['empty_answer']})",
          e["precision"] != e["precision"] and e["recall"] == 0 and e["f1"] == 0
          and e["strict_recall"] == 0 and e["empty_answer"] == 1)

    # unreadable answer: a parse failure, NOT an empty answer
    pf, _ = score_warnings([], gold, "gdpr", grounded=True, parse_ok=False)
    check(f"Parse failure counted apart from empty answers (parse_failed {pf['parse_failed']}, "
          f"empty {pf['empty_answer']})",
          pf["parse_failed"] == 1 and pf["empty_answer"] == 0 and pf["recall"] == 0)

    # definition echo: the model lists the Charter grounds as if they were columns
    from activity1 import CHARTER_GROUNDS
    echo_w, _ = parse_warnings(json.dumps({"sensitive_features": [
        {"feature": g, "ground": g, "reason": "r"} for g in CHARTER_GROUNDS]}), cols)
    ec, _ = score_warnings(echo_w, gold, "charter", grounded=True)
    check(f"Definition echo detected and graded as failed (echo {ec['definition_echo']}, "
          f"ground names {ec['echoed_grounds']}, recall {ec['recall']}, tp {ec['tp']})",
          ec["definition_echo"] == 1 and ec["recall"] == 0 and ec["tp"] == 0
          and ec["precision"] != ec["precision"])

# 6. Ollama running, model downloaded
try:
    listed = ollama.list()
    models = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
    names = [(m.get("model") or m.get("name")) if isinstance(m, dict) else (getattr(m, "model", None) or getattr(m, "name", ""))
             for m in models]
    check("Ollama is running", True)
    have = check(f"Model {MODEL} is downloaded", MODEL in names,
                 f"found {names}. Run: ollama pull {MODEL} (VPN on)")
except Exception as e:
    check("Ollama is running", False, f"{e!r}. Open the Ollama app.")
    have = False

# 7. One real call with forced JSON, including an enum (the ground field relies on it)
if have:
    from llm_fairness import make_ollama_llm
    llm = make_ollama_llm(MODEL)
    schema = {"type": "object",
              "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
              "required": ["answer"]}
    try:
        raw = llm('Is the sky blue? Reply with JSON: {"answer": "yes"} or {"answer": "no"}',
                  max_tokens=50, schema=schema)
        try:
            ok = json.loads(raw).get("answer") in ("yes", "no")
        except json.JSONDecodeError:
            ok = False
        check("Model answers with valid JSON that respects an enum", ok,
              f"got {raw[:80]!r}. Update: pip install -U ollama, and update the Ollama app.")
    except Exception as e:
        check("Model answers with valid JSON that respects an enum", False, f"{e!r}")

print(f"\n{sum(results)}/{len(results)} checks passed."
      + (" Ready: python run_sweep.py" if all(results) else " Fix the FAIL lines first."))