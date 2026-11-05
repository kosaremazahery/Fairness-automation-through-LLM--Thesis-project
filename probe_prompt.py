"""
probe_prompt.py — why does the model return an empty answer? A small diagnostic test.

Runs ONE prompt variant (default: gdpr) in four versions that each change ONE thing, and
shows which versions make the model find the sensitive columns. Nothing is logged to MLflow
and the sweep is not touched; raw answers are saved to probes/ for inspection.

  A  as-is         the current prompt, forced JSON                    (the baseline)
  B  no-escape     the line "If no column is sensitive, return an empty list." removed
  C  reworded      that line replaced by a version that asks to check every column first
  D  free-text     the current prompt, but WITHOUT forced JSON (the model writes freely;
                   the JSON is cut out of its text afterwards)

Reading the result:
  only B or C fixes it  -> the escape line is the cause (prompt wording)
  only D fixes it       -> forced JSON is the cause (output constraint, a code issue)
  nothing fixes it      -> the model is too small; keep the prompt, try a bigger model
  B fixes it, but adds false alarms on --variant minimal / charter -> prefer C

Usage (from the llm-fairness-compas folder, Ollama running):
    python probe_prompt.py                          # gdpr, all 4 versions, ~1-2 minutes
    python probe_prompt.py --variant charter        # same test on the Charter prompt (~10 min)
    python probe_prompt.py --variant minimal --versions A,C   # side effects of C (slow: ~20+ min)
    python probe_prompt.py --model llama3.1:8b      # another model
    python probe_prompt.py --versions A --max-tokens 1500   # repeat with the old length limit

Results are NOT logged to MLflow (on purpose: probes are diagnostics, not experiment runs).
They are printed, and saved with the exact prompts and raw answers in probes/*.json.
"""

import argparse
import json
import os
import time

import pandas as pd

from activity1 import GROUNDS_FOR, MAX_TOKENS, build_prompt, build_schema, parse_warnings
from llm_fairness import make_ollama_llm, runtime_info
from scoring import score_warnings

CSV = "compas-scores-two-years_v1.csv"
TARGET = "two_year_recid"
GOLD = os.path.join("gold", "compas.json")

ESCAPE_LINE = "- If no column is sensitive, return an empty list.\n"
# Worded so it fits all three prompts (the minimal prompt has no definition to check against),
# using the task's own words "directly records a sensitive characteristic".
REWORDED_LINE = ("- Check every column one by one. If, after checking all of them, none directly records "
                 "a sensitive characteristic, return an empty list.\n")

VERSIONS = {
    "A": ("as-is", lambda p: p, True),
    "B": ("no-escape", lambda p: p.replace(ESCAPE_LINE, ""), True),
    "C": ("reworded", lambda p: p.replace(ESCAPE_LINE, REWORDED_LINE), True),
    "D": ("free-text", lambda p: p, False),
}

# Same settings as the sweep: one deterministic answer + 5 sampled ones.
SETTINGS = [(0.0, 0)] + [(0.7, s) for s in range(5)]


def main():
    ap = argparse.ArgumentParser(description="Test prompt versions on one prompt variant.")
    ap.add_argument("--variant", default="gdpr", choices=list(GROUNDS_FOR))
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--versions", default="A,B,C,D", help="comma-separated subset, e.g. A,C")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help=f"answer length limit (default {MAX_TOKENS}; 1500 = the old limit)")
    args = ap.parse_args()
    versions = [v.strip().upper() for v in args.versions.split(",")]

    df = pd.read_csv(CSV).dropna(thresh=2)
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    basis = "charter" if args.variant == "minimal" else args.variant   # minimal: scored on the Charter key
    grounded = GROUNDS_FOR[args.variant] is not None
    positives = sorted(c["feature"] for c in gold["activity_1"]["columns"] if c[basis])

    base_prompt = build_prompt(df, TARGET, args.variant)
    if ESCAPE_LINE not in base_prompt:
        raise SystemExit("The escape line is not in the prompt any more (activity1.py was changed?). "
                         "Update ESCAPE_LINE in this script first.")

    runtime = runtime_info(args.model)
    print(f"Probe: prompt '{args.variant}' | model {args.model} | scored on the {basis} key "
          f"(sensitive: {positives})")
    print(f"Runtime: Ollama {runtime['ollama_version']} | model build {runtime['model_digest']} | "
          f"max tokens {args.max_tokens}\n")
    os.makedirs("probes", exist_ok=True)
    log, summary = [], []

    for v in versions:
        name, edit, use_schema = VERSIONS[v]
        prompt = edit(base_prompt)
        print(f"--- {v} {name}")
        rows = []
        for temperature, seed in SETTINGS:
            llm = make_ollama_llm(args.model, temperature=temperature, seed=seed)
            t0 = time.time()
            raw = llm(prompt, max_tokens=args.max_tokens, schema=build_schema(args.variant) if use_schema else None)
            secs = time.time() - t0
            warnings, ok = parse_warnings(raw, list(df.columns))
            m, d = score_warnings(warnings, gold, basis, grounded=grounded, parse_ok=ok)
            flagged = [w["feature"] for w in warnings]
            if not ok:
                what = ("PARSE FAILED (cut off at the length limit)" if not raw.rstrip().endswith("}")
                        else "PARSE FAILED (not valid JSON)")
            elif m["definition_echo"]:
                what = f"DEFINITION ECHO (listed {m['echoed_grounds']} grounds instead of columns)"
            else:
                what = flagged if flagged else "EMPTY"
            print(f"   temp {temperature} seed {seed}: {secs:5.1f}s | {what}")
            rows.append({"ok": ok, "empty": m["empty_answer"], "echo": m["definition_echo"],
                         "recall": m["recall"], "fp": m["fp"], "halluc": m["hallucinated"], "hits": d["hits"]})
            log.append({"version": v, "name": name, "temperature": temperature, "seed": seed,
                        "used_schema": use_schema, "max_tokens": args.max_tokens, **runtime,
                        "seconds": round(secs, 1), "flagged": flagged,
                        "raw_answer": raw, "prompt": prompt})
        r = pd.DataFrame(rows)
        found = {c: sum(c in h for h in r["hits"]) for c in positives}
        summary.append({
            "version": f"{v} {name}",
            "empty": f"{int(r['empty'].sum())}/{len(r)}",
            "parse_fail": int((~r["ok"]).sum()),
            "echo": int(r["echo"].sum()),
            "recall": round(r["recall"].mean(), 2),
            "false_alarms": round(r["fp"].mean(), 1),
            "hallucinated": round(r["halluc"].mean(), 1),
            "found": ", ".join(f"{c} {n}/{len(r)}" for c, n in found.items()),
        })
        print()

    print("Summary (6 runs per version: temp 0 seed 0 + temp 0.7 seeds 0-4; means over runs;\n"
          "empty = real empty answers only; parse failures and echoes are counted separately)")
    print(pd.DataFrame(summary).to_string(index=False))

    out = os.path.join("probes", f"probe_{args.variant}_{args.model.replace(':', '-')}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)
    print(f"\nRaw answers and exact prompts: {out}")


if __name__ == "__main__":
    main()