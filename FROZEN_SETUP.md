# Frozen setup — Activity 1 (sensitive features), frozen 2026-11-03

Everything below is fixed for the model comparison. Change nothing here between models;
if something must change, all runs are redone and the change is reported.

## Settings

| Setting | Value | Where | Why this value |
|---|---|---|---|
| Prompt wording | original ("A"), unchanged since the ground field was added | `activity1.py` | Reworded version C fixed GDPR on the 3B model but raised minimal-prompt false alarms from 3.7 to 17.5 (probe, 3 Nov). Kept A so the baseline is not distorted. |
| Prompt variants | minimal, charter, gdpr; only the definition paragraph (and the ground field) differs | `activity1.py` | Legal basis is the experimental condition |
| Ground field | closed list (JSON enum) for charter / gdpr; none for minimal | `activity1.py` | Strict scoring: right column AND right ground |
| Answer length limit | 3000 tokens | `activity1.MAX_TOKENS` | Enough to list all 53 COMPAS columns with a reason (~2,100 tokens). Chosen for that reason only, NOT tuned on scores. Note: 1500 vs 3000 changed the 3B GDPR answers (probe, 3 Nov). |
| Context window | 8192 tokens | `llm_fairness.py` | Prompt (~1,300) + answer (3,000) fit with room |
| Sampling | temperature 0 x seed 0, plus temperature 0.7 x seeds 0-4 | `run_sweep.SAMPLING` | One deterministic answer + 5 sampled ones to measure stability |
| Forced JSON | on (JSON schema) | `activity1.build_schema` | Free text was tested (probe version D) and changed nothing |
| Gold set | `gold/compas.json` version 1.1-draft (unchanged) | `gold/` | Not frozen legally: open questions on dob, colour, Art. 10 families |
| Data | `compas-scores-two-years_v1.csv`, 3 junk rows dropped -> 7,214 rows | `run_sweep.run_one` | |
| First model | llama3.2:3b, then llama3.1:8b with the same settings | `run_sweep.MODELS` | |

Recorded automatically with every run (MLflow params): prompt_sha, max_tokens, temperature,
seed, gold_version, ollama_version, model_digest, git_commit, plus a copy of the code files.

## Prompt fingerprints (COMPAS)

Every MLflow run stores `prompt_sha`. Runs of the frozen setup have exactly these:

| Variant | prompt_sha |
|---|---|
| minimal | dcd7a3b1111b |
| charter | 5b7d042b1ae5 |
| gdpr | 90c977a3aa0f |

The exact texts are in `prompts/activity1_prompt_<variant>.txt`.

## Files

| File | Job | Fingerprint (SHA-256, first 12) |
|---|---|---|
| activity1.py | builds the prompts, calls the model, reads the answer | 88c68ca17d23 |
| scoring.py | grades an answer (lenient, strict, failed answers, error families) | 1960305b701d |
| run_sweep.py | runs all model x prompt x seed combinations into MLflow | bb9919b3a688 |
| rescore.py | re-grades saved answers; prints the summary tables | b5e5541e8783 |
| llm_fairness.py | talks to Ollama; records Ollama version and model build | e21547946949 |
| check_setup.py | checks files, data, gold set, scoring, Ollama (24 checks) | 1e649701c0a9 |
| show_reasons.py | prints the model's reasons for the latest runs | 65a5a507a9b7 |
| probe_prompt.py | prompt diagnostics (not part of the experiment) | 85319b4bef5c |

To check a file on Windows (PowerShell): `Get-FileHash activity1.py` — the first 12 characters
(lower case) must match the table.

## Run order

1. `python check_setup.py` -> 24/24
2. `python run_sweep.py` -> 18 runs for llama3.2:3b (about 1 hour)
3. `python rescore.py` -> summary tables
4. add `"llama3.1:8b"` to `MODELS` in `run_sweep.py`, then repeat 2-3
