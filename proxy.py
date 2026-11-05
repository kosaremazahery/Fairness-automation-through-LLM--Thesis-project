"""
proxy.py — step 2 of activity 1: proxies for sensitive attributes.

A proxy is a column that does not record a sensitive attribute but lets you guess it
(e.g. a first name revealing ethnic origin). Two parts:

  python proxy.py measure       the DATA side: which columns are measurable proxies?
                                Per-column cross-validated AUC for predicting each sensitive
                                attribute, plus one dataset-level sAUC (Ceccon et al. 2025).
                                Writes the measured proxy gold set: gold/compas_proxies.json.
                                No model calls; runs in about a minute.

  python proxy.py ask           the LLM side: which columns does the LLM THINK are proxies?
                                Same models and seeds as run_sweep.py (MODELS, SAMPLING), logged
                                to the MLflow experiment 'llm-fairness-proxies'. Finished runs are
                                skipped, so it can be stopped and restarted.
                                --models llama3.2:3b   to run one model only

  python proxy.py report        scores every saved LLM answer against the measured gold set and
                                prints: precision/recall per attribute and threshold, flags outside
                                the candidate columns, and per column how often the model flagged
                                it next to its measured AUC.

How the LLM answer is scored: for each attribute (race, sex, age), the candidate columns the
model flags are compared with the measured proxies (AUC >= 0.70; also at 0.60). Flags on the
sensitive columns themselves, on excluded columns (outcomes, scores, identifiers, dates) and on
names that are not columns are counted apart, not as false alarms, because the gold set has no
label for them. The model is told which columns are sensitive (step 1's gold answer), so step 2
is measured on its own, without step-1 errors carried over.

All rules below are fixed BEFORE looking at any result and are saved into the output file,
so every label can be traced to a written rule. The labels are "statistical proxies in this
dataset", not legal judgements: whether using such a column is indirect discrimination needs
an objective-justification test, which is out of scope.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

CSV = "compas-scores-two-years_v1.csv"
GOLD = os.path.join("gold", "compas.json")
OUT = os.path.join("gold", "compas_proxies.json")
LIT = os.path.join("gold", "compas_proxies_literature.json")      # literature-based key (hand-built, sourced)
VERSION = "p1.0-draft"

# ------------------------------------------------------------------ fixed rules
RULES = {
    "protected_attributes": {
        "race": {
            "primary": "African-American vs Caucasian (rows of other groups left out of this comparison). "
                       "Same contrast as Ceccon et al. 2025 and as 53 of 59 COMPAS papers in Simson et al. 2024.",
            "secondary": "all 6 race groups, one-vs-rest AUC averaged over groups (macro), all rows",
        },
        "sex": {"primary": "Male vs Female, all rows"},
        "age": {
            "primary": "under 25 vs 25 and over, all rows (the contrast highlighted by ProPublica)",
            "secondary": "the 3 age_cat groups, one-vs-rest macro AUC",
        },
    },
    "candidate_columns": "Only columns that could be a legitimate model input: gold families "
                         "'name', 'criminal_history', 'current_charge'. Excluded by rule: the sensitive "
                         "columns themselves, the target, re-offence outcomes (recidivism_outcome), COMPAS "
                         "model outputs (risk_score), identifiers, process dates.",
    "candidate_families": ["name", "criminal_history", "current_charge"],
    "encoding": {
        "numeric": "the value itself, gradient-boosted trees (HistGradientBoostingClassifier, defaults)",
        "categorical (<= 50 distinct values)": "one-hot + logistic regression",
        "text (> 50 distinct values, e.g. names, charge descriptions)": "character 2-4-grams (TF-IDF, char_wb) + logistic regression",
    },
    "max_categories_for_one_hot": 50,
    "validation": "5-fold stratified cross-validation, shuffled, random_state=0; AUC of the out-of-fold "
                  "predicted probabilities (so high-cardinality columns cannot score by memorising rows)",
    "threshold_primary": 0.70,
    "threshold_primary_source": "AUC 0.70-0.80 = 'acceptable discrimination' (Hosmer & Lemeshow, Applied Logistic Regression)",
    "threshold_secondary": 0.60,
    "dataset_level": "sAUC: all candidate columns together, same encodings, logistic regression on the combined features",
}
SEED = 0
FOLDS = 5


def load():
    df = pd.read_csv(CSV).dropna(thresh=2)
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    fam = {c["feature"]: c.get("family") for c in gold["activity_1"]["columns"]}
    cands = [c for c in df.columns if fam.get(c) in RULES["candidate_families"]]
    return df, gold, cands


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical" if s.nunique(dropna=True) <= RULES["max_categories_for_one_hot"] else "text"


def model_for(kind: str):
    if kind == "numeric":
        return HistGradientBoostingClassifier(random_state=SEED)
    if kind == "categorical":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             LogisticRegression(max_iter=2000))
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
                         LogisticRegression(max_iter=2000))


def prepare(s: pd.Series, kind: str):
    if kind == "numeric":
        return s.to_frame()
    if kind == "categorical":
        return s.fillna("missing").astype(str).to_frame()
    return s.fillna("").astype(str).str.lower()          # 1-D input for the text vectoriser


def oof_auc(X, y, model) -> float:
    """Out-of-fold AUC. y: binary 0/1, or multi-class labels (macro one-vs-rest)."""
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def targets(df: pd.DataFrame) -> dict:
    """attribute -> {variant: (row mask, labels)}; built exactly as RULES describes."""
    race2 = df["race"].isin(["African-American", "Caucasian"])
    return {
        "race": {"primary": (race2, (df.loc[race2, "race"] == "African-American").astype(int)),
                 "secondary": (pd.Series(True, index=df.index), df["race"].astype(str))},
        "sex": {"primary": (pd.Series(True, index=df.index), (df["sex"] == "Female").astype(int))},
        "age": {"primary": (pd.Series(True, index=df.index), (df["age"] < 25).astype(int)),
                "secondary": (pd.Series(True, index=df.index), df["age_cat"].astype(str))},
    }


def sauc(df, cands, mask, y) -> float:
    kinds = {c: column_kind(df[c]) for c in cands}
    parts = []
    for c, k in kinds.items():
        if k == "numeric":
            parts.append((c, make_pipeline(FunctionTransformer(lambda a: np.nan_to_num(np.asarray(a, float))),
                                           StandardScaler()), [c]))
        elif k == "categorical":
            parts.append((c, make_pipeline(FunctionTransformer(lambda a: pd.DataFrame(a).fillna("missing").astype(str)),
                                           OneHotEncoder(handle_unknown="ignore")), [c]))
        else:
            parts.append((c, make_pipeline(FunctionTransformer(lambda a: pd.Series(np.asarray(a).ravel()).fillna("").astype(str).str.lower()),
                                           TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)), [c]))
    model = make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=3000))
    return oof_auc(df.loc[mask, cands], y, model)


def measure():
    df, gold, cands = load()
    print(f"Candidate columns ({len(cands)}): {cands}\n")
    t1, t2 = RULES["threshold_primary"], RULES["threshold_secondary"]
    result = {"dataset": gold["dataset"], "version": VERSION, "gold_version": gold["version"],
              "rows": len(df), "rules": RULES, "attributes": {}}
    for attr, variants in targets(df).items():
        mask, y = variants["primary"]
        n_pos = int(y.sum())
        print(f"=== {attr}: {RULES['protected_attributes'][attr]['primary']}  "
              f"(rows {int(mask.sum())}, positive class {n_pos})")
        rows = []
        for c in cands:
            kind = column_kind(df[c])
            auc = oof_auc(prepare(df.loc[mask, c], kind), y, model_for(kind))
            row = {"feature": c, "kind": kind, "auc": round(auc, 3),
                   "proxy": auc >= t1, f"proxy_at_{t2}": auc >= t2}
            if "secondary" in variants:
                m2, y2 = variants["secondary"]
                row["auc_secondary"] = round(oof_auc(prepare(df.loc[m2, c], kind), y2, model_for(kind)), 3)
            rows.append(row)
        rows.sort(key=lambda r: -r["auc"])
        s = round(sauc(df, cands, mask, y), 3)
        result["attributes"][attr] = {"definition": RULES["protected_attributes"][attr],
                                      "sauc": s, "columns": rows,
                                      "proxies": [r["feature"] for r in rows if r["proxy"]],
                                      f"proxies_at_{t2}": [r["feature"] for r in rows if r[f"proxy_at_{t2}"]]}
        for r in rows:
            sec = f"   (all groups: {r['auc_secondary']:.3f})" if "auc_secondary" in r else ""
            flag = "PROXY" if r["proxy"] else ("weak " if r[f"proxy_at_{t2}"] else "")
            print(f"   {r['feature']:<16} {r['kind']:<11} AUC {r['auc']:.3f}  {flag:<5}{sec}")
        print(f"   sAUC (all candidates together): {s:.3f}\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Measured proxy gold set ({VERSION}) written to {OUT}")
    print(f"PROXY = AUC >= {t1} (primary threshold); weak = AUC >= {t2} (secondary).")


# ================================================================== the LLM side
PROXY_EXPERIMENT = "llm-fairness-proxies"
ATTRS = ["race", "sex", "age"]
SENSITIVE_GIVEN = ["race", "sex", "age", "age_cat", "dob"]     # step 1's answer, given to the model

PROXY_PROMPT = """You are auditing a tabular dataset for fairness risks before it is used to train a machine learning model.

The model will predict the column '{target}'.

Columns (name | type | example values):
{columns}

The columns {sensitive} directly record sensitive characteristics of a person and will not be used as model inputs.

Task: identify the OTHER columns that could reveal a person's race, sex or age, even though they do not record it directly (proxy variables).
- For each proxy, say which characteristic it could reveal: "race", "sex" or "age". A column may reveal more than one; list it once per characteristic.
- Use column names exactly as written above.
- Give a one-sentence reason for each.
- If no column is a proxy, return an empty list.

Respond only with JSON of the form:
{{"proxies": [{{"feature": "<column name>", "reveals": "race" | "sex" | "age", "reason": "<one sentence>"}}]}}
"""

PROXY_SCHEMA = {
    "type": "object",
    "properties": {"proxies": {"type": "array", "items": {
        "type": "object",
        "properties": {"feature": {"type": "string"}, "reveals": {"type": "string", "enum": ATTRS},
                       "reason": {"type": "string"}},
        "required": ["feature", "reveals", "reason"]}}},
    "required": ["proxies"],
}


def build_proxy_prompt(df, target="two_year_recid"):
    from activity1 import describe_columns
    return PROXY_PROMPT.format(target=target, columns=describe_columns(df),
                               sensitive=", ".join(f"'{c}'" for c in SENSITIVE_GIVEN))


def parse_proxies(raw, columns):
    """-> (list of {feature, reveals, reason, in_dataset}, parse_ok). Never raises."""
    import re
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict) or not isinstance(data.get("proxies"), list):
        return [], False
    lookup = {c.lower().strip(): c for c in columns}
    out, seen = [], set()
    for item in data["proxies"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("feature", "")).strip().strip("`'\"")
        attr = str(item.get("reveals", "")).strip().lower()
        if not name or attr not in ATTRS:
            continue
        feature = lookup.get(name.lower(), name)
        if (feature, attr) in seen:
            continue
        seen.add((feature, attr))
        out.append({"feature": feature, "reveals": attr, "reason": str(item.get("reason", "")),
                    "in_dataset": feature in columns})
    return out, True


def load_literature():
    """Literature key -> {'literature_identity': {attr: set}, 'literature_any': {attr: set}, 'labels': {(f, attr): label}}."""
    with open(LIT, encoding="utf-8") as f:
        lit = json.load(f)
    identity = {a: set() for a in ATTRS}
    any_concern = {a: set() for a in ATTRS}
    labels = {}
    for e in lit["entries"]:
        labels[(e["feature"], e["attribute"])] = e["label"]
        if e["label"] == "proxy":
            identity[e["attribute"]].add(e["feature"])
        if e["label"] in ("proxy", "contested", "correlated", "tainted"):
            any_concern[e["attribute"]].add(e["feature"])
    return {"version": lit["version"], "literature_identity": identity, "literature_any": any_concern, "labels": labels}


def score_proxies(answer, proxy_gold, all_columns, parse_ok=True, level="proxies", gold_sets=None):
    """Per attribute: flagged candidate columns vs a reference set of proxies.
    Reference = measured proxies (level = 'proxies' for AUC >= 0.70, 'proxies_at_0.6' for >= 0.60),
    or, if gold_sets is given ({attr: set of columns}), that set (e.g. the literature key).
    Flags outside the candidate set are counted apart: on sensitive columns, on excluded columns,
    or on names that are not columns."""
    nan = float("nan")
    metrics, detail = {}, {}
    for attr in ATTRS:
        a = proxy_gold["attributes"][attr]
        cands = {r["feature"] for r in a["columns"]}
        auc = {r["feature"]: r["auc"] for r in a["columns"]}
        gold = set(gold_sets[attr]) & cands if gold_sets is not None else set(a[level])
        flagged = {p["feature"] for p in answer if p["reveals"] == attr}
        inside = flagged & cands
        tp, fp, fn = inside & gold, inside - gold, gold - inside
        prec = len(tp) / len(inside) if inside else nan
        rec = len(tp) / len(gold) if gold else nan
        if rec != rec:
            f1 = nan
        elif rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec) if prec == prec and (prec + rec) else nan
        m = {"tp": len(tp), "fp": len(fp), "fn": len(fn), "precision": prec, "recall": rec, "f1": f1,
             "n_gold": len(gold),
             "flag_sensitive": len(flagged & set(SENSITIVE_GIVEN)),
             "flag_excluded": len((flagged & set(all_columns)) - cands - set(SENSITIVE_GIVEN)),
             "flag_invented": len(flagged - set(all_columns)),
             # does the model pick the columns that leak most? mean AUC of flagged vs unflagged candidates
             "mean_auc_flagged": float(np.mean([auc[c] for c in inside])) if inside else nan,
             "mean_auc_unflagged": float(np.mean([auc[c] for c in cands - inside])) if cands - inside else nan}
        metrics.update({f"{attr}_{k}": v for k, v in m.items()})
        detail[attr] = {"hits": sorted(tp), "false_alarms": sorted(fp), "misses": sorted(fn),
                        "outside_candidates": sorted(flagged - cands)}
    metrics["empty_answer"] = int(parse_ok and not answer)
    metrics["parse_failed"] = int(not parse_ok)
    return metrics, detail


def ask(models, sampling):
    import time
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from activity1 import MAX_TOKENS, prompt_fingerprint
    from llm_fairness import make_ollama_llm, runtime_info
    from run_sweep import TRACKING_URI

    if not os.path.exists(OUT):
        raise SystemExit(f"{OUT} not found. Run 'python proxy.py measure' first.")
    with open(OUT, encoding="utf-8") as f:
        proxy_gold = json.load(f)
    df, gold, _ = load()
    prompt = build_proxy_prompt(df)
    sha = prompt_fingerprint(prompt)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(PROXY_EXPERIMENT)

    pairs = []
    for temperature, seeds in sampling:
        pairs += [(temperature, s) for s in (seeds[:1] if temperature == 0 else seeds)]
    grid = [(m, t, s) for m in models for t, s in pairs]
    print(f"{len(grid)} proxy runs planned (prompt {sha}, proxy gold {proxy_gold['version']})\n")
    for i, (model, temperature, seed) in enumerate(grid, 1):
        label = f"[{i}/{len(grid)}] {model} | temp {temperature} | seed {seed}"
        done = mlflow.search_runs(
            experiment_names=[PROXY_EXPERIMENT], max_results=1,
            filter_string=(f"tags.status = 'ok' and params.model = '{model}' and params.temperature = '{temperature}' "
                           f"and params.seed = '{seed}' and params.prompt_sha = '{sha}' "
                           f"and params.max_tokens = '{MAX_TOKENS}'"))
        if not done.empty:
            print(f"{label}  (already done, skipped)")
            continue
        print(f"→ {label}")
        call_log = []
        llm = make_ollama_llm(model, temperature=temperature, seed=seed, call_log=call_log)
        with mlflow.start_run(run_name=f"proxies|{model}|t{temperature}|s{seed}"):
            mlflow.log_params({"activity": "proxies", "dataset": gold["dataset"], "model": model,
                               "temperature": temperature, "seed": seed, "max_tokens": MAX_TOKENS,
                               "prompt_sha": sha, "proxy_gold_version": proxy_gold["version"],
                               **runtime_info(model)})
            t0 = time.time()
            try:
                raw = llm(prompt, max_tokens=MAX_TOKENS, schema=PROXY_SCHEMA)
            except Exception as e:
                mlflow.set_tag("status", "crashed")
                print(f"   CRASHED: {e!r}")
                continue
            finally:
                mlflow.log_metric("runtime_s", time.time() - t0)
                mlflow.log_dict(call_log, "llm_calls.json")
            answer, ok = parse_proxies(raw, list(df.columns))
            mlflow.set_tags({"status": "ok", "parse_ok": str(ok)})
            mlflow.log_dict(answer, "proxies.json")
            metrics, detail = score_proxies(answer, proxy_gold, list(df.columns), parse_ok=ok)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(detail, "score_proxies.json")
            pairs_txt = [f"{p['feature']}->{p['reveals']}" for p in answer]
            print(f"   ok in {time.time() - t0:.1f}s | {pairs_txt if ok else 'PARSE FAILED'}")
            for attr in ATTRS:
                d = detail[attr]
                print(f"   [{attr}] hits={d['hits']} false_alarms={d['false_alarms']} misses={d['misses']} "
                      f"outside={d['outside_candidates']}")


def report():
    """Re-grade every saved proxy answer against the current measured gold and summarise."""
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from run_sweep import TRACKING_URI
    with open(OUT, encoding="utf-8") as f:
        proxy_gold = json.load(f)
    lit = load_literature()
    # the four references every answer is scored against
    references = [("measured AUC>=0.70", {"level": "proxies"}),
                  ("measured AUC>=0.60", {"level": "proxies_at_0.6"}),
                  ("literature: identity", {"gold_sets": lit["literature_identity"]}),
                  ("literature: any concern", {"gold_sets": lit["literature_any"]})]
    df, _, _ = load()
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[PROXY_EXPERIMENT], filter_string="tags.status = 'ok'")
    if runs.empty:
        raise SystemExit("No proxy runs yet. Run 'python proxy.py ask' first.")
    rows, flags = [], []
    for _, run in runs.iterrows():
        answer = mlflow.artifacts.load_dict(f"runs:/{run['run_id']}/proxies.json")
        ok = run.get("tags.parse_ok") != "False"
        base = {"model": run["params.model"], "temperature": run["params.temperature"], "seed": run["params.seed"]}
        for ref_name, kwargs in references:
            m, _ = score_proxies(answer, proxy_gold, list(df.columns), parse_ok=ok, **kwargs)
            for attr in ATTRS:
                rows.append({**base, "reference": ref_name, "attribute": attr,
                             **{k: m[f"{attr}_{k}"] for k in ("precision", "recall", "f1", "fp", "n_gold",
                                                              "flag_sensitive", "flag_excluded", "flag_invented",
                                                              "mean_auc_flagged", "mean_auc_unflagged")},
                             "empty": m["empty_answer"], "parse_fail": m["parse_failed"]})
        for p in answer:
            flags.append({**base, "feature": p["feature"], "attribute": p["reveals"]})
    t = pd.DataFrame(rows)
    os.makedirs("rescores", exist_ok=True)
    t.to_csv(os.path.join("rescores", f"proxies_{proxy_gold['version']}.csv"), index=False)
    keys = ["model", "temperature", "attribute", "reference"]
    order = {name: i for i, (name, _) in enumerate(references)}
    t = t.sort_values(by=["model", "temperature", "attribute", "reference"],
                      key=lambda s: s.map(order) if s.name == "reference" else s)
    g = t.groupby(keys, sort=False)
    summary = g[["precision", "recall", "f1", "fp", "n_gold"]].agg(
        {"precision": "mean", "recall": "mean", "f1": "mean", "fp": "mean", "n_gold": "first"}).round(2)
    summary["runs"] = g.size()
    print(f"Proxy scores (mean over runs) against four references: measured key {proxy_gold['version']} at two "
          f"thresholds, literature key {lit['version']} (identity proxies only / any documented concern).\n"
          "n_gold = reference proxies for that attribute; precision '-' = no candidate flagged; "
          "recall '-' = the reference has no proxy for that attribute.\n")
    print(summary.to_string(na_rep="-"))

    # reference-independent columns, once per model x temperature x attribute
    one = t[t["reference"] == references[0][0]].groupby(["model", "temperature", "attribute"])
    extra = one[["flag_sensitive", "flag_excluded", "flag_invented", "mean_auc_flagged", "mean_auc_unflagged",
                 "empty", "parse_fail"]].mean().round(2)
    print("\nFlags outside the candidate columns (mean per run) and leakage of the flagged columns:\n"
          "  flag_sensitive = the sensitive columns themselves; flag_excluded = outcomes, scores, identifiers, dates;\n"
          "  flag_invented = names that are not columns; mean_auc_* = measured AUC of flagged vs unflagged candidates\n")
    print(extra.to_string(na_rep="-"))

    # per-column view: how often the model flagged each candidate, next to its measured AUC
    f = pd.DataFrame(flags)
    n_runs = runs.groupby(["params.model", "params.temperature"]).size()
    print("\nPer column: measured AUC [literature label] : share of runs that flagged it (temperature 0.7 runs)\n"
          "  labels: P = proxy, C = contested, R = correlated, T = tainted, - = not discussed")
    short = {"proxy": "P", "contested": "C", "correlated": "R", "tainted": "T"}
    for attr in ATTRS:
        cols = proxy_gold["attributes"][attr]["columns"]
        print(f"\n  {attr}")
        for model in sorted(t["model"].unique()):
            n = int(n_runs.get((model, "0.7"), 0))
            if not n:
                continue
            sub = f[(f["model"] == model) & (f["temperature"] == "0.7") & (f["attribute"] == attr)] if not f.empty else f
            share = sub["feature"].value_counts() / n if not sub.empty else pd.Series(dtype=float)
            line = ", ".join(f"{c['feature']} {c['auc']:.2f}[{short.get(lit['labels'].get((c['feature'], attr)), '-')}]"
                             f":{share.get(c['feature'], 0):.0%}" for c in cols)
            print(f"    {model:<13} {line}")
    print(f"\nPer-run details: rescores/proxies_{proxy_gold['version']}.csv")


def main():
    ap = argparse.ArgumentParser(description="Proxy step: measure proxies in the data, ask the LLM, report.")
    ap.add_argument("command", choices=["measure", "ask", "report"])
    ap.add_argument("--models", default=None,
                    help="comma-separated models for 'ask' (default: MODELS in run_sweep.py)")
    args = ap.parse_args()
    if args.command == "measure":
        measure()
    elif args.command == "ask":
        from run_sweep import MODELS, SAMPLING
        ask(args.models.split(",") if args.models else MODELS, SAMPLING)
    else:
        report()


if __name__ == "__main__":
    main()
