"""
import_probes.py — load probe runs that used the FROZEN setup into MLflow, without calling a model.

Why: probe_prompt.py version A runs with a 3000-token limit are exactly the runs run_sweep.py
would make (same prompt, same limit, same temperature and seeds). Re-running them would cost an
hour and give the same kind of answers. This script logs them as ordinary sweep runs, so that
rescore.py and run_sweep.py see them; run_sweep.py then skips them as already done.

What is imported, and what is not:
  imported  answers whose prompt is EXACTLY a frozen prompt (same prompt_sha as activity1.py
            builds today) and that ran with the frozen answer limit (activity1.MAX_TOKENS)
  skipped   any other version (B, C, D), runs with another token limit (e.g. the 1500 test),
            and runs already in MLflow
  Probe files written before the token limit was recorded have no "max_tokens" field. Those are
  SKIPPED unless you pass --assume-max-tokens, and then only the files you name are trusted:
  never use a wildcard together with --assume-max-tokens, because an older file with another
  limit (e.g. the 31 Oct probe, 1500 tokens) would be imported as if it had the frozen one.

Every imported run is tagged source=probe_import and keeps the probe file name, so the thesis
can say exactly where each result came from.

Usage (from the llm-fairness-compas folder):
    python import_probes.py probes\\*.json                  # files that recorded their limit
    python import_probes.py probes\\probe_gdpr_llama3.2-3b_20261103-1539.json --assume-max-tokens 3000
    python import_probes.py <files> --reset                  # first delete all earlier imports
"""

import argparse
import glob
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from activity1 import BASES_FOR, GROUNDS_FOR, MAX_TOKENS, build_prompt, parse_warnings, prompt_fingerprint
from run_sweep import DATASETS, EXPERIMENT_NAME, TRACKING_URI, already_done, log_provenance
from scoring import score_warnings

DATASET = "compas"


def main():
    ap = argparse.ArgumentParser(description="Import frozen-setup probe runs into MLflow.")
    ap.add_argument("files", nargs="+", help="probe JSON files (wildcards allowed)")
    ap.add_argument("--model", default="llama3.2:3b", help="model the probe ran on (default llama3.2:3b)")
    ap.add_argument("--assume-max-tokens", type=int, default=None,
                    help="token limit for old probe files that did not record it (name those files "
                         "explicitly; no wildcard)")
    ap.add_argument("--reset", action="store_true",
                    help="delete every run imported earlier (tag source=probe_import) before importing")
    args = ap.parse_args()

    files = sorted({f for pattern in args.files for f in glob.glob(pattern)})
    if not files:
        raise SystemExit("No probe files found. Run from the llm-fairness-compas folder.")

    cfg = DATASETS[DATASET]
    df = pd.read_csv(cfg["path"]).dropna(thresh=2)
    frozen = {prompt_fingerprint(build_prompt(df, cfg["target_col"], v)): v for v in GROUNDS_FOR}
    gold_path = os.path.join("gold", f"{DATASET}.json")
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    if args.reset:
        old = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                                 filter_string="tags.source = 'probe_import'")
        for run_id in old["run_id"] if not old.empty else []:
            mlflow.delete_run(run_id)
        print(f"Reset: deleted {len(old)} previously imported run(s).\n")
    imported, skipped = 0, {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in files:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            sha = prompt_fingerprint(e["prompt"])
            variant = frozen.get(sha)
            if variant is None:
                skip(f"not a frozen prompt (version {e.get('version')} {e.get('name')})")
                continue
            if not e.get("used_schema", True):
                skip("ran without forced JSON")
                continue
            max_tokens = e.get("max_tokens", args.assume_max_tokens)
            if max_tokens is None:
                skip("token limit not recorded in the file (see --assume-max-tokens)")
                continue
            if max_tokens != MAX_TOKENS:
                skip(f"token limit {max_tokens}, frozen is {MAX_TOKENS}")
                continue
            temperature, seed = float(e["temperature"]), int(e["seed"])
            if already_done(DATASET, cfg, args.model, variant, temperature, seed):
                skip("already in MLflow")
                continue

            warnings, parse_ok = parse_warnings(e["raw_answer"], list(df.columns))
            grounded = GROUNDS_FOR[variant] is not None
            with mlflow.start_run(run_name=f"{DATASET}|{args.model}|{variant}|t{temperature}|s{seed}"):
                mlflow.log_params({
                    "activity": "sensitive_features",
                    "dataset": DATASET,
                    "target_col": cfg["target_col"],
                    "model": args.model,
                    "provider": "ollama",
                    "prompt_variant": variant,
                    "output_schema": "grounded" if grounded else "plain",
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "gold_version": gold.get("version", "?"),
                    "ollama_version": e.get("ollama_version", "unknown"),
                    "model_digest": e.get("model_digest", "unknown"),
                    "prompt_sha": sha,
                })
                log_provenance(gold_path)
                mlflow.set_tags({"status": "ok", "parse_ok": str(parse_ok),
                                 "source": "probe_import", "probe_file": os.path.basename(path)})
                mlflow.log_metric("runtime_s", float(e.get("seconds", 0)))
                mlflow.log_metric("n_flagged", len(warnings))
                mlflow.log_dict(warnings, "warnings.json")
                mlflow.log_dict([{"prompt": e["prompt"], "response": e["raw_answer"],
                                  "used_schema": True}], "llm_calls.json")
                for basis in BASES_FOR[variant]:
                    metrics, detail = score_warnings(warnings, gold, basis, grounded=grounded, parse_ok=parse_ok)
                    mlflow.log_metrics({f"{basis}_{k}": v for k, v in metrics.items()})
                    mlflow.log_dict(detail, f"score_{basis}.json")
            imported += 1
            print(f"imported  {variant:<8} temp {temperature} seed {seed}  ({os.path.basename(path)})")

    print(f"\nImported {imported} run(s) into MLflow.")
    for reason, n in skipped.items():
        print(f"Skipped {n}: {reason}")
    print("\nNext: python rescore.py  (run_sweep.py will now skip the imported runs)")


if __name__ == "__main__":
    main()