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
| Gold set | `gold/compas.json` version 1.2-draft (labels unchanged since 1.0-draft) | `gold/` | dob decided 4 Nov (follows age); open: colour, Art. 10 families |
| Data | `compas-scores-two-years_v1.csv`, 3 junk rows dropped -> 7,214 rows | `run_sweep.run_one` | Not in the public repository (personal records): download from ProPublica, see README |
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

The exact texts are in `prompts/activity1_prompt_<variant>.txt`. That folder is kept out of the public repository, because the prompts show three example values per column, including real defendants' names. Anyone with the CSV can rebuild them with `activity1.build_prompt`; the fingerprints above confirm the text is identical.

## Files

| File | Job | Fingerprint (SHA-256, first 12) |
|---|---|---|
| activity1.py | builds the prompts, calls the model, reads the answer | 88c68ca17d23 |
| scoring.py | grades an answer (lenient, strict, failed answers, error families) | 1960305b701d |
| run_sweep.py | runs all model x prompt x seed combinations into MLflow (MODELS = 3B and 8B) | 89668af2e352 |
| rescore.py | re-grades saved answers; frozen-setup runs only (`--all` every run; `--key charter` shared key) | ac5e72486ea6 |
| llm_fairness.py | talks to Ollama; records Ollama version and model build | e21547946949 |
| check_setup.py | checks files, data, gold sets, scoring, scikit-learn, Ollama (25 checks) | 990c70eb4de5 |
| show_reasons.py | prints the model's reasons for the latest runs | 65a5a507a9b7 |
| probe_prompt.py | prompt diagnostics (not part of the experiment) | a30043ccf9a4 |
| import_probes.py | loads frozen-setup probe runs into MLflow (no model calls) | e16678da27e4 |
| proxy.py | step 2: measured proxies, LLM proxies, report against both keys | 84afc00be38e |
| gold/compas_proxies_literature.json | literature proxy key L1.0-draft, with sources | 0a8a919c5425 |

run_sweep.py's fingerprint changes whenever MODELS is edited (adding a model is the intended change); all other files must match. To check a file on Windows (PowerShell): `Get-FileHash activity1.py` — the first 12 characters
(lower case) must match the table.

## Run order

1. `python check_setup.py` -> 24/24
2. llama3.2:3b: the 18 frozen-setup runs already exist as probe version A (3 Nov, 3000 tokens).
   They are in probes\ files 20261103-1539 (gdpr), -1549 (charter), -1636 (minimal), which do not
   record their token limit, so they are named explicitly:
   `python import_probes.py <those three files> --assume-max-tokens 3000 --reset`
   (tagged source=probe_import; versions B/C/D are skipped automatically). Never use a wildcard
   with --assume-max-tokens: the 31 Oct probe file also lacks the field but ran with 1500.
3. `python run_sweep.py` -> runs only what is missing (nothing, if step 2 imported all 18)
4. `python rescore.py` -> summary tables
5. add `"llama3.1:8b"` to `MODELS` in `run_sweep.py`, then repeat 3-4
