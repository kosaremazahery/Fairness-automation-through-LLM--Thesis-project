"""
rescore.py — re-grade saved LLM answers against the current gold set, without calling any model.

Why: run_sweep.py grades each run against the gold file (and scoring code) as it was on the
day of the run. When the gold set or the scoring changes, the stored scores become outdated.
Every run already saved the model's answer as warnings.json, so we only need to grade again.

What it does:
  1. loads gold/<dataset>.json and reads its "version"
  2. finds all successful MLflow runs for that dataset
  3. downloads each run's warnings.json (the model's saved answer)
  4. scores it with scoring.score_warnings() — the same function run_sweep.py uses
  5. writes one CSV with every run x legal basis, and prints two summary tables:
     the scores (lenient + strict), and the false alarms per column family.
     Rows are grouped by model, prompt, basis AND temperature, so the deterministic
     answer (temperature 0) is never averaged together with the sampled ones.

Reading the summary:
  precision  mean over runs that gave a real answer; '-'/NaN if no run did
  empty      share of runs where the model flagged nothing (0.4 = 2 of 5 runs)
  parse_fail share of runs whose answer was not readable JSON (often cut off at the length limit)
  echo       share of runs where the model listed the definition's grounds instead of columns
  *_sd       standard deviation over seeds: how much the answer changes between runs

Which runs are graded:
  By default only runs of the FROZEN setup (see FROZEN_SETUP.md): the exact prompt text that
  activity1.py builds today (same prompt_sha) and the frozen answer limit (MAX_TOKENS). Older
  runs (other wording, 1500-token limit, old variants) stay in MLflow but are not mixed into
  the tables. --all grades everything and adds max_tokens as a column, to compare setups.

Usage (from the llm-fairness-compas folder):
    python rescore.py                         # gold/compas.json, frozen-setup runs only
    python rescore.py --all                   # every run, grouped by token limit as well
    python rescore.py --key charter           # all prompts graded against the Charter key (compare definitions)
    python rescore.py --gold gold/adult.json  # another dataset
    python rescore.py --log                   # also store the new scores inside each MLflow run

What it cannot do: if the PROMPT or the output schema changes, the saved answers do not
contain the new information. That needs new runs, not re-grading. (Example: runs made before
the "ground" field existed have no grounds, so their strict scores stay empty.)
"""

import argparse
import json
import os
import re

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from activity1 import BASES_FOR, MAX_TOKENS, build_prompt, prompt_fingerprint
from scoring import score_warnings

TRACKING_URI = "sqlite:///mlflow.db"          # same as run_sweep.py
EXPERIMENT_NAME = "llm-fairness-warnings"


def safe_version(version: str) -> str:
    """'1.0-draft' -> '1.0-draft'; anything MLflow would reject in a metric name becomes '_'."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(version))


def load_answer(run_id: str):
    """The model's saved answer for one run, or None if the run has no warnings.json."""
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/warnings.json")
    except Exception:
        return None


def check_gold(gold: dict, path: str) -> set:
    """Stops with a readable message if the gold file is not in the binary charter/gdpr format.
    Returns the legal bases the file labels."""
    problems = []
    if "version" not in gold:
        problems.append('no "version" field')
    columns = gold.get("activity_1", {}).get("columns", [])
    if not columns:
        problems.append('no "activity_1" -> "columns" list')
    known = {b for b in ("charter", "gdpr") if columns and all(isinstance(c.get(b), bool) for c in columns)}
    if columns and not known:
        problems.append('columns have no true/false "charter" or "gdpr" keys '
                        f"(first column has keys: {sorted(columns[0])})")
    if problems:
        raise SystemExit(
            f"\n{path} looks like an OLD gold file: " + "; ".join(problems) + ".\n"
            "Replace it with the current gold/compas.json (version 1.0-draft or later), then run again.")
    if "families" not in gold["activity_1"]:
        print(f"Note: {path} has no column families, so false alarms will not be typed (fp_<family> = 0).\n")
    return known


def main():
    parser = argparse.ArgumentParser(description="Re-grade saved runs against the current gold set.")
    parser.add_argument("--gold", default=os.path.join("gold", "compas.json"),
                        help="gold file to grade against (default: gold/compas.json)")
    parser.add_argument("--out", default="rescores",
                        help="folder for the output CSV (default: rescores/)")
    parser.add_argument("--log", action="store_true",
                        help="also write the new scores into each MLflow run, as g<version>_<basis>_<metric>")
    parser.add_argument("--key", choices=["charter", "gdpr"], default=None,
                        help="grade EVERY prompt against this one answer key (shared yardstick), "
                             "e.g. --key charter to compare minimal, charter and gdpr prompts directly")
    parser.add_argument("--all", action="store_true",
                        help="grade every run, not only the frozen setup (adds max_tokens to the tables)")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    known_bases = check_gold(gold, args.gold)
    dataset = gold["dataset"]
    version = safe_version(gold["version"])

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.status = 'ok' and params.dataset = '{dataset}'",
    )
    if runs.empty:
        raise SystemExit(f"No successful runs for dataset '{dataset}'. Run run_sweep.py first.")

    # The frozen prompts' fingerprints, built from the same data the sweep uses.
    frozen_sha = {}
    if not args.all:
        from run_sweep import DATASETS
        if dataset not in DATASETS:
            raise SystemExit(f"'{dataset}' is not in run_sweep.DATASETS, so its frozen prompts are unknown. "
                             "Add it there, or use --all.")
        cfg = DATASETS[dataset]
        df = pd.read_csv(cfg["path"]).dropna(thresh=2)
        frozen_sha = {v: prompt_fingerprint(build_prompt(df, cfg["target_col"], v)) for v in BASES_FOR}

    print(f"Re-grading runs of '{dataset}' against gold version {version} "
          f"({'all runs' if args.all else f'frozen setup only: frozen prompts, {MAX_TOKENS}-token limit'})\n")
    rows, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        variant = run.get("params.prompt_variant")
        if variant not in BASES_FOR:
            skip(f"old prompt variant {variant!r}")
            continue
        if not args.all:
            limit = run.get("params.max_tokens")
            if limit != str(MAX_TOKENS):
                shown = "not recorded (run from before 3 Nov)" if pd.isna(limit) else limit
                skip(f"not the frozen setup: token limit {shown}")
                continue
            if run.get("params.prompt_sha") != frozen_sha[variant]:
                skip("not the frozen setup: different prompt wording")
                continue
        answer = load_answer(run_id)
        if answer is None:
            skip("no warnings.json saved (made by an older run_sweep.py)")
            continue
        # Strict scoring only for runs whose prompt asked for grounds. Runs from before the
        # ground field existed have no output_schema param, so they stay lenient-only.
        grounded = run.get("params.output_schema") == "grounded"
        # run_sweep stores parse_ok as a tag; runs without the tag are treated as parsed.
        parse_ok = run.get("tags.parse_ok") != "False"

        # --key: every prompt graded against the same answer key. Strict (ground) scoring only
        # where the key is the prompt's own definition: GDPR ground names cannot match Charter grounds.
        bases = [args.key] if args.key else BASES_FOR[variant]
        for basis in bases:
            if basis not in known_bases:
                continue
            strict_ok = grounded and basis in BASES_FOR[variant] and variant != "minimal"
            metrics, detail = score_warnings(answer, gold, basis, grounded=strict_ok, parse_ok=parse_ok)
            rows.append({
                "run_id": run_id,
                "start_time": run["start_time"],
                "model": run.get("params.model"),
                "prompt": variant,
                "temperature": run.get("params.temperature"),
                "seed": run.get("params.seed"),
                "max_tokens": run.get("params.max_tokens"),
                "source": run.get("tags.source") or "sweep",
                "prompt_sha": run.get("params.prompt_sha"),
                "basis": basis,
                "graded_with": version,
                "originally_graded_with": run.get("params.gold_version"),
                **metrics,
                "missed_columns": ";".join(detail["misses"]),
                "false_alarm_columns": ";".join(detail["false_alarms"]),
                "hallucinated_names": ";".join(detail["hallucinated"]),
                "wrong_ground_columns": ";".join(f"{k}={v}" for k, v in detail.get("wrong_ground", {}).items()),
            })
            if args.log:
                for k, v in metrics.items():
                    client.log_metric(run_id, f"g{version}_{basis}_{k}", v)
                client.log_dict(run_id, detail, f"rescore/g{version}/score_{basis}.json")

    if skipped:
        print(f"Not graded: {sum(skipped.values())} run(s)")
        for reason, n in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"  {n:>3}  {reason}")
        print()

    if not rows:
        raise SystemExit("Nothing to grade: every run was skipped (see the reasons above). "
                         "Do a fresh run with run_sweep.py, then try again.")

    table = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{dataset}_gold_{version}" + (f"_key-{args.key}" if args.key else "") + ".csv")
    table.to_csv(out_path, index=False)

    keys = ["model", "prompt", "basis", "temperature"] + (["max_tokens"] if args.all else [])
    table["temperature"] = table["temperature"].fillna("?")
    table["max_tokens"] = table["max_tokens"].fillna("?")
    score_cols = ["precision", "recall", "f1", "strict_precision", "strict_recall", "wrong_ground",
                  "off_definition", "fp", "hallucinated", "empty_answer", "parse_failed", "definition_echo"]
    score_cols = [c for c in score_cols if c in table.columns]
    grouped = table.groupby(keys)
    summary = grouped[score_cols].mean().round(2).rename(
        columns={"empty_answer": "empty", "parse_failed": "parse_fail", "definition_echo": "echo"})
    # spread over seeds: how much recall and F1 change from one run to the next
    for c in ("recall", "f1"):
        summary[f"{c}_sd"] = grouped[c].std().round(2)
    summary["runs"] = grouped.size()
    if args.key:
        print(f"SHARED KEY: every prompt graded against the '{args.key}' answer key "
              f"(strict scores only for the '{args.key}' prompt itself).\n")
    print("Scores (mean over runs; precision skips failed answers; empty / parse_fail / echo = share\n"
          "of runs with that kind of failed answer; strict_* NaN = prompt had no ground field;\n"
          "*_sd NaN = only one run)")
    print(summary.to_string(na_rep="-"))

    fam_cols = sorted(c for c in table.columns if c.startswith("fp_") and c != "fp_art10")
    if fam_cols:
        fam = grouped[fam_cols + ["fp_art10"]].mean().round(1)
        fam.columns = [c[3:] for c in fam.columns]
        print("\nFalse alarms by column family (mean over runs; art10 = criminal-offence data, GDPR Art. 10)")
        print(fam.to_string())

    print(f"\nPer-run details (misses, false alarms, wrong grounds, hallucinations): {out_path}")
    if args.log:
        print(f"Scores also stored in each MLflow run as g{version}_<basis>_<metric>.")


if __name__ == "__main__":
    main()
