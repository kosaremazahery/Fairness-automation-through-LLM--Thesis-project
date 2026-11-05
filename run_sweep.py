"""
run_sweep.py — experiment sweep with MLflow tracking (activity 1, step 1: sensitive features).

One MLflow *run* = one combination of (dataset, model, prompt_variant, temperature, seed).
  - params    : the knobs you set (model, prompt variant, temperature, seed, prompt fingerprint, ...)
  - tags      : status, whether the model's JSON could be parsed
  - metrics   : per legal basis: lenient (tp, fp, fn, precision, recall, f1), strict
                (right column AND right ground; charter/gdpr prompts only), false alarms
                per column family (fp_<family>, fp_art10), failed answers (empty_answer,
                parse_failed, definition_echo); plus runtime
  - artifacts : warnings.json, llm_calls.json (prompt + raw answer), score_<basis>.json, error.txt

Scoring (scoring.py) is binary (sensitive / non-sensitive) against gold/<dataset>.json.
The minimal prompt is scored against both legal bases; each legal prompt against its own.

Temperature and seeds (see SAMPLING below)
  temperature 0   : the model always picks its most likely next word, so the answer is
                    fixed. Extra seeds would only repeat the same answer -> one seed only.
  temperature > 0 : the model samples, so answers can differ. The seed fixes the random
                    draws, so each (seed, temperature) answer is reproducible, and several
                    seeds show how STABLE the model's judgement is.

View results:   mlflow ui --backend-store-uri sqlite:///mlflow.db
then open http://127.0.0.1:5000
Summary tables: python rescore.py
"""

import itertools
import json
import os
import time
import traceback

# MLflow tries to record the git commit of each run. Without git installed this
# only prints long warnings, so silence them.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from activity1 import (BASES_FOR, GROUNDS_FOR, MAX_TOKENS, build_prompt, detect_sensitive_features,
                       prompt_fingerprint)
from llm_fairness import make_ollama_llm, runtime_info
from scoring import score_warnings  # noqa: F401  (re-exported: rescore/check_setup used to import it from here)

# ---------------------------------------------------------------- config
TRACKING_URI = "sqlite:///mlflow.db"          # local, works fully offline
EXPERIMENT_NAME = "llm-fairness-warnings"

DATASETS = {
    "compas": {"path": "compas-scores-two-years_v1.csv", "target_col": "two_year_recid"},
    # "adult": {"path": "adult.csv", "target_col": "class"},
}
MODELS = ["llama3.2:3b", "llama3.1:8b"]                       # must match a name from `ollama list`
PROMPT_VARIANTS = ["minimal", "charter", "gdpr"]

# (temperature, seeds). The first line is the deterministic baseline (one answer);
# the second gives 5 sampled answers per model x prompt, to measure stability.
SAMPLING = [
    (0.0, [0]),
    (0.7, [0, 1, 2, 3, 4]),
]

# Skip (dataset, model, prompt, temperature, seed) combinations that already have a
# successful run, so you can stop the sweep and restart it, or add a model, without
# repeating finished work. Set to False to force everything to run again.
SKIP_DONE = True


# ---------------------------------------------------------------- code provenance
# Files copied into every run (artifacts/code/), so each result can be traced to the exact
# code and gold file that produced it, even if they were never committed to git.
CODE_FILES = ["activity1.py", "scoring.py", "run_sweep.py", "llm_fairness.py"]


def git_state():
    """(commit, dirty) of the project folder; ('no-git', None) if git or the repo is missing."""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "no-git", None


def log_provenance(gold_path):
    commit, dirty = git_state()
    mlflow.log_params({"git_commit": commit, "git_dirty": dirty})
    for f in CODE_FILES + ([gold_path] if os.path.exists(gold_path) else []):
        if os.path.exists(f):
            mlflow.log_artifact(f, artifact_path="code/" + os.path.dirname(f) if os.path.dirname(f) else "code")


def fmt(x):
    """0.57 -> '0.57'; NaN (undefined, e.g. precision of an empty answer) -> '  - '."""
    return f"{x:.2f}" if x == x else "  - "


def sampling_grid():
    """Expands SAMPLING into (temperature, seed) pairs. At temperature 0 every seed gives
    the same answer, so only the first seed is kept (with a warning)."""
    pairs = []
    for temperature, seeds in SAMPLING:
        if temperature == 0 and len(seeds) > 1:
            print(f"NOTE: temperature 0 is deterministic; using only seed {seeds[0]} of {seeds}.\n")
            seeds = seeds[:1]
        pairs += [(temperature, s) for s in seeds]
    return pairs


def already_done(dataset_name, cfg, model, variant, temperature, seed):
    """True if a successful run exists with the same settings, the exact same prompt text
    (same prompt_sha) and the same answer length limit. Runs made with an older prompt wording
    or the old 1500-token limit therefore do not count, and are run again."""
    df = pd.read_csv(cfg["path"]).dropna(thresh=2)
    sha = prompt_fingerprint(build_prompt(df, cfg["target_col"], variant))
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(f"tags.status = 'ok' and params.dataset = '{dataset_name}' "
                       f"and params.model = '{model}' and params.prompt_variant = '{variant}' "
                       f"and params.temperature = '{temperature}' and params.seed = '{seed}' "
                       f"and params.prompt_sha = '{sha}' and params.max_tokens = '{MAX_TOKENS}'"),
        max_results=1,
    )
    return not runs.empty


# ---------------------------------------------------------------- one run
def run_one(dataset_name, cfg, model, variant, temperature, seed):
    # The _v1 CSV ends with 3 junk rows holding a single value (decile_score.1 = 4.5096,
    # apparently a leftover mean). Drop rows with fewer than 2 non-empty values.
    df = pd.read_csv(cfg["path"]).dropna(thresh=2)
    call_log = []
    llm = make_ollama_llm(model, temperature=temperature, seed=seed, call_log=call_log)
    grounded = GROUNDS_FOR[variant] is not None

    gold_path = os.path.join("gold", f"{dataset_name}.json")
    gold = None
    if os.path.exists(gold_path):
        with open(gold_path, encoding="utf-8") as f:
            gold = json.load(f)

    with mlflow.start_run(run_name=f"{dataset_name}|{model}|{variant}|t{temperature}|s{seed}"):
        mlflow.log_params({
            "activity": "sensitive_features",
            "dataset": dataset_name,
            "target_col": cfg["target_col"],
            "model": model,
            "provider": "ollama",
            "prompt_variant": variant,
            "output_schema": "grounded" if grounded else "plain",
            "temperature": temperature,
            "seed": seed,
            "max_tokens": MAX_TOKENS,
            "gold_version": gold.get("version", "?") if gold else "none",
            # same name + prompt + seed can still answer differently after an Ollama update
            # or a re-pulled model, so the exact runtime is part of every run's record
            **runtime_info(model),
        })
        log_provenance(gold_path)

        t0 = time.time()
        try:
            warnings, parse_ok, prompt = detect_sensitive_features(df, cfg["target_col"], llm, variant)
        except Exception as e:
            mlflow.set_tag("status", "crashed")
            mlflow.log_text(traceback.format_exc(), "error.txt")
            print(f"   CRASHED after {time.time() - t0:.1f}s: {e!r}")
            return
        finally:
            # Always keep what the model actually said, crash or not.
            mlflow.log_metric("runtime_s", time.time() - t0)
            mlflow.log_dict(call_log, "llm_calls.json")

        mlflow.log_param("prompt_sha", prompt_fingerprint(prompt))
        mlflow.set_tag("status", "ok")
        mlflow.set_tag("parse_ok", str(parse_ok))
        mlflow.log_metric("n_flagged", len(warnings))
        mlflow.log_dict(warnings, "warnings.json")

        flagged = [w["feature"] for w in warnings]
        if not parse_ok:
            raw = call_log[-1]["response"] if call_log else ""
            what = ("PARSE FAILED: answer cut off at the length limit" if not raw.rstrip().endswith("}")
                    else "PARSE FAILED: answer is not valid JSON")
        else:
            what = f"flagged={flagged}" if flagged else "flagged=NOTHING (empty answer)"
        print(f"   ok in {time.time() - t0:.1f}s | {what}")
        if gold:
            for basis in BASES_FOR[variant]:
                metrics, detail = score_warnings(warnings, gold, basis, grounded=grounded, parse_ok=parse_ok)
                if metrics["definition_echo"]:
                    print(f"   [{basis}] DEFINITION ECHO: the model listed grounds instead of columns "
                          f"({metrics['echoed_grounds']} ground names). Graded as a failed answer.")
                mlflow.log_metrics({f"{basis}_{k}": v for k, v in metrics.items()})
                mlflow.log_dict(detail, f"score_{basis}.json")
                line = (f"   [{basis}] P={fmt(metrics['precision'])} R={fmt(metrics['recall'])} "
                        f"F1={fmt(metrics['f1'])}")
                if "strict_f1" in metrics:
                    line += (f" | strict P={fmt(metrics['strict_precision'])} R={fmt(metrics['strict_recall'])}"
                             f" wrong_ground={detail['wrong_ground']}")
                print(line)
                print(f"      misses={detail['misses']} false_alarms={metrics['fp']} "
                      f"(art10={metrics['fp_art10']}) by family={ {k: len(v) for k, v in detail['false_alarms_by_family'].items()} } "
                      f"hallucinated={detail['hallucinated']}")


# ---------------------------------------------------------------- sweep
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    commit, dirty = git_state()
    if commit == "no-git":
        print("WARNING: not a git repository (or git not installed). Runs keep a code copy, "
              "but no commit hash.\n")
    elif dirty:
        print(f"WARNING: uncommitted changes on top of {commit}. Runs keep a code copy, but "
              "commit first if these results matter.\n")
    else:
        print(f"Code at commit {commit} (clean).\n")

    pairs = sampling_grid()
    grid = list(itertools.product(DATASETS.items(), MODELS, PROMPT_VARIANTS, pairs))
    print(f"{len(grid)} runs planned: {len(MODELS)} model(s) x {len(PROMPT_VARIANTS)} prompts x "
          f"{len(pairs)} (temperature, seed) settings x {len(DATASETS)} dataset(s)\n")
    for i, ((name, cfg), model, variant, (temperature, seed)) in enumerate(grid, 1):
        label = f"[{i}/{len(grid)}] {name} | {model} | {variant} | temp {temperature} | seed {seed}"
        if SKIP_DONE and already_done(name, cfg, model, variant, temperature, seed):
            print(f"{label}  (already done, skipped)")
            continue
        print(f"→ {label}")
        run_one(name, cfg, model, variant, temperature, seed)


if __name__ == "__main__":
    main()
