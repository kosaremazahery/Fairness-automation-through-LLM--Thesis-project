# LLM fairness automation: sensitive features and proxies

Work in progress, MSc thesis, expected March 2027. Pilot results on COMPAS below; Adult dataset and larger models in progress.

Master's thesis project (University of Bologna) on whether, and how reliably, a large language
model can automate a fairness check on a tabular dataset: given the dataset's columns, which ones
are **sensitive features**, and which ones are **proxies** for them?

The system takes a dataset and returns a list of **warnings** for a human expert to check.
Each part of the system is scored against a gold set, across several models, prompts and random
seeds. Every run is logged with MLflow.

## Research questions

1. **Sensitive features.** Does the model flag the columns that directly record a sensitive
   characteristic (race, sex, age, ...)? Does it follow the definition of "sensitive" it is given,
   or fall back on its own idea of it?
2. **Proxies.** Do the model's guesses about proxy columns match the proxies measured in the data,
   and what the literature says about them?
3. **Reliability.** How stable are the answers across seeds, and how do they change with model size?

The definition given to the model is the list of grounds in Article 21 of the EU Charter of
Fundamental Rights, used only as a *source* of the definition. The project makes **no claim that
the labels are legally correct**.

## How it works

```
dataset (CSV) ──► step 1: sensitive features ──► step 2: proxies ──► warnings
                    activity1.py + run_sweep.py     proxy.py
                    scored by scoring.py            measured key (AUC) + literature key
```

**Step 1: sensitive features.** The model sees each column's name, type and 3 example values, and
returns the columns that directly record a sensitive characteristic, as JSON. There are three prompt
variants that differ only in the definition paragraph:

| Variant | Definition given | Answer format |
|---|---|---|
| `minimal` | none | column + reason |
| `charter` | EU Charter Art. 21 grounds | column + ground (from a closed list) + reason |
| `gdpr` | GDPR Art. 9 special categories (kept for comparison, set aside) | column + ground + reason |

Answers are scored per column against `gold/compas.json`:
- **lenient:** right column
- **strict:** right column and right ground
- **failed answers**, counted separately: empty, cut off, or a copy of the definition instead of columns
- **false alarms by type of column**

**Step 2: proxies.** A proxy is a column that does not record a sensitive characteristic but lets
you guess it (e.g. a first name revealing sex).
- **Measured key:** `proxy.py measure` computes, for each candidate column, the cross-validated
  AUC of predicting race, sex or age from that column alone (threshold 0.70; 0.60 also reported).
- **Literature key:** `gold/compas_proxies_literature.json` records, with sources, what the
  literature says about each column. It separates *proxies* (the column reveals who someone is)
  from *tainted features* (the column's values were shaped by past discrimination, e.g. arrest
  records).
- **LLM side:** `proxy.py ask` asks the model which columns could reveal race, sex or age, and
  `proxy.py report` scores its answers against both keys.

## Setup

Requirements: Python 3.11+, [Ollama](https://ollama.com) running locally.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull llama3.2:3b             # and llama3.1:8b for the second model
```

**Data.** The COMPAS file is not in this repository, because it contains personal records (names,
dates of birth, criminal histories). Download `compas-scores-two-years.csv` from
[ProPublica's compas-analysis repository](https://github.com/propublica/compas-analysis), save it in
this folder as `compas-scores-two-years_v1.csv`. For Adult, save the UCI Adult data (train and test
combined, 48,842 rows, with a header row; the common Kaggle/OpenML `adult.csv`) in this folder as
`adult.csv`. Then run:

```bash
python check_setup.py               # must end with "28/28 checks passed"
```

## Running

| Command | What it does | Time (laptop, CPU) |
|---|---|---|
| `python check_setup.py` | checks files, data, gold sets, scoring and Ollama | seconds |
| `python run_sweep.py` | step 1: every model × prompt × seed; skips runs already done | 3B: ~1 h, 8B: ~5 h |
| `python rescore.py` | re-grades saved answers; prints the summary tables | seconds |
| `python rescore.py --key charter` | grades every prompt against the same (Charter) key | seconds |
| `python show_reasons.py` | prints the model's reasons for its latest flags | seconds |
| `python proxy.py measure` | step 2, data side: measured proxy key | ~1 min |
| `python proxy.py ask --models llama3.2:3b` | step 2, LLM side | 3B: minutes |
| `python proxy.py report` | step 2 scores against both keys | seconds |
| `python probe_prompt.py` | diagnostic: tests prompt variants (not part of the experiment) | minutes |
| `python import_probes.py <files> --assume-max-tokens 3000` | loads probe runs of the frozen setup into MLflow | seconds |

View all runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://127.0.0.1:5000.

## Frozen setup

The experiment settings are fixed and documented in [`FROZEN_SETUP.md`](FROZEN_SETUP.md):
- prompt wording A
- answer limit 3000 tokens
- temperature 0 once, plus temperature 0.7 with seeds 0–4
- gold set 1.2-draft

Every run records its prompt fingerprint, token limit, Ollama version and model build. Change
nothing between models; if something must change, rerun everything and report it.

The token limit is frozen because a pilot showed that **changing it alone (1500 → 3000) flipped
the 3B model's answers** to the GDPR prompt.

## Repository layout

```
activity1.py          step 1: prompts, model call, answer parsing
scoring.py            step 1: grading (lenient, strict, failed answers, false-alarm types)
run_sweep.py          step 1: experiment sweep with MLflow
rescore.py            re-grading of saved answers, summary tables
proxy.py              step 2: measured proxies, LLM proxies, report
llm_fairness.py       Ollama client; records Ollama version and model build
check_setup.py        pre-flight checks
show_reasons.py       model reasons for the latest runs
probe_prompt.py       prompt diagnostics
import_probes.py      imports frozen-setup probe runs into MLflow
FROZEN_SETUP.md       frozen settings and file fingerprints
gold/
  compas.json                       step 1 gold set (1.2-draft): sensitive columns, grounds, column families
  compas_proxies.json               measured proxy key (written by proxy.py measure)
  compas_proxies_literature.json    literature proxy key, with sources (L1.0-draft)
rescores/             result tables written by rescore.py and proxy.py report (no personal data)
```

Kept out of the repository on purpose (see `.gitignore`):
- the COMPAS CSV (personal records; download it from ProPublica, see Setup)
- `prompts/` and `probes/`: the prompts show example values from the data, including real names
- the MLflow run history

Anyone with the CSV can rebuild the exact prompts with `activity1.build_prompt`. The prompt
fingerprints in `FROZEN_SETUP.md` confirm the text is identical.

**Earlier prototype, not used in the experiments:** `main.py`, `pipeline.py`, `metrics_module.py`
(an end-to-end prototype in which the LLM picks an attribute, groups and metrics, then explains
them), `agent.py`, `agent_prompt.py`, `tools.py`, `tool_registry.py` (an agent that calls fairness
tools in a loop), and the `Dockerfile` (which runs `main.py`).

## Results and run history

- **Public:** the result tables in `rescores/` (scores per run; no personal data).
- **Private:** the full run history, in `mlflow.db`, `mlruns/` and `mlartifacts/`. It contains every
  prompt and raw answer, and therefore real names from the data, so it is excluded by `.gitignore`.
  **Back it up privately**, e.g. zip it to a personal cloud drive after each sweep.

## Status (November 2026)

| Part | Status |
|---|---|
| Step 1 on COMPAS, llama3.2:3b and llama3.1:8b | done |
| Measured proxy key (COMPAS) | done |
| Literature proxy key (COMPAS) | draft |
| Proxy step: llama3.2:3b and llama3.1:8b | done |
| Adult gold set (step 1) | draft |
| Step 1 on Adult, llama3.2:3b | running |
| Adult proxy step | to do |
| Larger models (e.g. through OpenRouter) | to decide with supervisors |
| "Tainted features" as a second warning type | to decide with supervisors |

## Main pilot findings so far

- **Reliability:** the 8B model is more reliable than the 3B one (0 vs 3 failed answers in
  18 runs), but not more accurate.
- **Age:** both models find race and sex, and almost never find age. This matches the usual
  treatment of COMPAS in the fairness literature.
- **Proxies:** the names are the only measured proxies (race: AUC 0.72–0.78; sex: 0.90–0.94). The
  3B model never flags them; its few flags match the literature's *tainted* columns instead. The 8B
  model starts to find the names for race (precision 1.00, recall 0.27) but never flags the first
  name for sex, the strongest proxy, and flags many columns outside the candidate set.

## References

Key sources are listed in the project's literature notes and in
`gold/compas_proxies_literature.json`. Data: ProPublica's COMPAS analysis (Angwin et al., 2016).