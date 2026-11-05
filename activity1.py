"""
activity1.py — Ciatto's activity 1, step 1: select sensitive features.

Input : a DataFrame (+ the name of the target column)
Output: a list of warnings, one per flagged column:
        {"activity": "sensitive_features", "feature": ..., "ground": ... or None,
         "message": ..., "in_dataset": bool}

Step 1 asks only for columns that DIRECTLY record a sensitive characteristic.
Columns that merely correlate with one (proxies) are step 2.

Prompt variants (the only thing that changes between them is the definition paragraph
and, for the legal variants, the extra "ground" field):
  minimal  : no definition of "sensitive", no ground field
  charter  : EU Charter of Fundamental Rights, Art. 21(1)-(2); ground from the Charter list
  gdpr     : GDPR, Art. 9(1) special categories; ground from the Art. 9(1) list

Design choices (worth stating in the thesis):
  * The model sees column names, dtypes and 3 example values. No dataset-specific
    hints (no "race", no group names) are written into the prompt.
  * Output is forced to JSON with a schema, so parsing failures measure the model,
    not our string-splitting.
  * For the legal variants the "ground" is a CLOSED list (a JSON-schema enum), taken
    from the same Python list that writes the definition paragraph, so the two can
    never drift apart. One extra value, OTHER_GROUND, lets the model say it is flagging
    a column that records none of the listed grounds. That is an explicit,
    self-reported step outside the definition, and it is counted separately.
  * Flagged names are NOT filtered against the real columns. A name that does not
    exist is kept and scored as a hallucination.
"""

import hashlib
import json
import re

import pandas as pd

ACTIVITY = "sensitive_features"
OTHER_GROUND = "none of the listed grounds"

# Longest answer the model may write, in tokens. 1500 was too short: answers that list almost
# every column with a reason were cut off mid-JSON and became parse failures, hiding what the
# model actually said. 3000 lets them finish, so they are scored as false alarms instead.
MAX_TOKENS = 3000

# Grounds in the order and wording of the legal texts (see the "Answer Keys v1" doc).
CHARTER_GROUNDS = [
    "sex", "race", "colour", "ethnic or social origin", "genetic features", "language",
    "religion or belief", "political or any other opinion", "membership of a national minority",
    "property", "birth", "disability", "age", "sexual orientation", "nationality",
]
GDPR_GROUNDS = [
    "racial or ethnic origin", "political opinions", "religious or philosophical beliefs",
    "trade union membership", "genetic data", "biometric data for uniquely identifying a person",
    "data concerning health", "data concerning a person's sex life or sexual orientation",
]


def _oxford_list(items):
    return ", ".join(items[:-1]) + ", or " + items[-1]


CHARTER = (
    "Definition: a column is a sensitive feature if it directly records one of the grounds of "
    "discrimination listed in Article 21 of the EU Charter of Fundamental Rights: "
    + _oxford_list(CHARTER_GROUNDS) + "."
)
GDPR = (
    "Definition: a column is a sensitive feature if it directly records one of the special categories "
    "of personal data listed in Article 9(1) of the GDPR: "
    + _oxford_list(GDPR_GROUNDS) + "."
)

VARIANTS = {
    "minimal": "",
    "charter": CHARTER + "\n\n",
    "gdpr": GDPR + "\n\n",
}

# Closed list of grounds the model may answer with, per variant (None = no ground field).
GROUNDS_FOR = {
    "minimal": None,
    "charter": CHARTER_GROUNDS + [OTHER_GROUND],
    "gdpr": GDPR_GROUNDS + [OTHER_GROUND],
}

# Which answer key(s) each prompt variant is scored against.
BASES_FOR = {"minimal": ["charter", "gdpr"], "charter": ["charter"], "gdpr": ["gdpr"]}

PROMPT = """You are auditing a tabular dataset for fairness risks before it is used to train a machine learning model.

The model will predict the column '{target}'.

Columns (name | type | example values):
{columns}

{definition}Task: identify every column that directly records a sensitive characteristic of a person.
- Do not include columns that are only correlated with a sensitive characteristic.
- Use column names exactly as written above.
{ground_rule}- Give a one-sentence reason for each.
- If no column is sensitive, return an empty list.

Respond only with JSON of the form:
{output_format}
"""

GROUND_RULE = (
    "- For each column, give the ground from the definition that it records, written exactly as in "
    "the definition. If you flag a column that records none of the listed grounds, write "
    f"\"{OTHER_GROUND}\".\n"
)
FORMAT_PLAIN = '{"sensitive_features": [{"feature": "<column name>", "reason": "<one sentence>"}]}'
FORMAT_GROUNDED = ('{"sensitive_features": [{"feature": "<column name>", "ground": "<ground from the definition>", '
                   '"reason": "<one sentence>"}]}')


def build_schema(variant: str) -> dict:
    """JSON schema passed to the model. Legal variants get an enum-constrained 'ground'."""
    item = {
        "type": "object",
        "properties": {"feature": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["feature", "reason"],
    }
    grounds = GROUNDS_FOR[variant]
    if grounds:
        item["properties"] = {
            "feature": {"type": "string"},
            "ground": {"type": "string", "enum": grounds},
            "reason": {"type": "string"},
        }
        item["required"] = ["feature", "ground", "reason"]
    return {
        "type": "object",
        "properties": {"sensitive_features": {"type": "array", "items": item}},
        "required": ["sensitive_features"],
    }


# Kept for backward compatibility with code that imports SCHEMA (the minimal schema).
SCHEMA = build_schema("minimal")


def describe_columns(df: pd.DataFrame, n_examples: int = 3) -> str:
    lines = []
    for col in df.columns:
        examples = df[col].dropna().astype(str).unique()[:n_examples]
        ex = ", ".join(e[:30] for e in examples)
        lines.append(f"- {col} | {df[col].dtype} | {ex}")
    return "\n".join(lines)


def build_prompt(df: pd.DataFrame, target_col: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}; choose from {list(VARIANTS)}")
    grounded = GROUNDS_FOR[variant] is not None
    return PROMPT.format(
        target=target_col,
        columns=describe_columns(df),
        definition=VARIANTS[variant],
        ground_rule=GROUND_RULE if grounded else "",
        output_format=FORMAT_GROUNDED if grounded else FORMAT_PLAIN,
    )


def prompt_fingerprint(prompt: str) -> str:
    """Short hash of the exact prompt text, logged so every result traces to its wording."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def parse_warnings(raw: str, columns: list) -> tuple:
    """Returns (warnings, parse_ok). Never raises."""
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)          # JSON wrapped in prose / code fences
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict) or not isinstance(data.get("sensitive_features"), list):
        return [], False

    lookup = {c.lower().strip(): c for c in columns}
    warnings, seen = [], set()
    for item in data["sensitive_features"]:
        ground = None
        if isinstance(item, str):
            name, reason = item, ""
        elif isinstance(item, dict):
            name, reason = str(item.get("feature", "")), str(item.get("reason", ""))
            if item.get("ground") is not None:
                ground = str(item["ground"]).strip()
        else:
            continue
        clean = name.strip().strip("`'\"")
        if not clean:
            continue
        feature = lookup.get(clean.lower(), clean)   # fix case/spacing, keep unknown names
        if feature in seen:
            continue
        seen.add(feature)
        warnings.append({
            "activity": ACTIVITY,
            "feature": feature,
            "ground": ground,
            "message": reason,
            "in_dataset": feature in columns,
        })
    return warnings, True


def detect_sensitive_features(df: pd.DataFrame, target_col: str, llm, variant: str = "minimal"):
    """Returns (warnings, parse_ok, prompt)."""
    prompt = build_prompt(df, target_col, variant)
    raw = llm(prompt, max_tokens=MAX_TOKENS, schema=build_schema(variant))
    warnings, ok = parse_warnings(raw, list(df.columns))
    return warnings, ok, prompt
