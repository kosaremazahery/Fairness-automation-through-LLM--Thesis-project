"""
scoring.py — grading of activity 1 (sensitive features) against one legal basis.

Used by run_sweep.py (fresh runs), rescore.py (re-grading saved answers) and probe_prompt.py,
so all three always grade the same way.

Two levels of correctness
  lenient : right column                      (tp, fp, fn, precision, recall, f1)
  strict  : right column AND right ground     (strict_tp, strict_precision, ...)
            Only computed when the prompt asked for grounds, i.e. for the 'charter' and
            'gdpr' prompts. The 'minimal' prompt has no ground field, so its strict
            metrics are simply absent (NaN in rescore's CSV), never zero.

Three kinds of failed answer, counted separately (each 0 or 1 per run)
  empty_answer    the model answered correctly formatted JSON, but flagged nothing
  parse_failed    the answer could not be read as JSON (often: cut off at the length limit)
  definition_echo the model copied the definition's list of grounds instead of naming columns
                  (at least ECHO_MIN invented "column" names that are ground names, e.g.
                  'colour', 'religion or belief', 'nationality' on COMPAS)
  For all three, precision is UNDEFINED (NaN): there are no real flags whose quality could be
  judged. recall and f1 are 0: every sensitive column was missed. An echo answer is graded as
  failed even if some ground names happen to equal column names ('sex', 'race', 'age' on
  COMPAS): the model did not look at the columns, it repeated the definition.
  pandas' mean() skips NaN, so averaged precision describes only the real answers, and the
  three counts say how often a run produced no real answer, and why.

Error typing of false alarms
  Every non-sensitive column in the gold file has a "family" (identifier, name,
  criminal_history, ...). False alarms are counted per family (fp_<family>), and
  families marked "gdpr_art10" are summed into fp_art10: criminal-offence data,
  which GDPR treats under Art. 10, not as an Art. 9 special category. A high fp_art10
  under the GDPR prompt is the "Art. 10 confusion". fp_protected counts Charter-only
  grounds (sex, age, ...) flagged under the GDPR key: the model carried the broader
  notion of "sensitive" into the narrower definition.
"""

from collections import Counter

from activity1 import CHARTER_GROUNDS, GDPR_GROUNDS

ACTIVITY = "sensitive_features"
OTHER_GROUND = "none of the listed grounds"
NAN = float("nan")

# An answer is a "definition echo" if at least this many flagged names are not columns but
# ground names from a definition. One or two can be coincidence (a model inventing a
# 'nationality' column); the echo seen on COMPAS had 12.
ECHO_MIN = 3


def _norm(s):
    return " ".join(str(s).lower().replace("’", "'").split()) if s is not None else ""


GROUND_NAMES = {_norm(g) for g in CHARTER_GROUNDS + GDPR_GROUNDS}


def _f1(p, r):
    """F1 from precision and recall. Undefined precision with zero recall still gives 0:
    nothing was found. Undefined only when recall is undefined too."""
    if r != r:                       # recall NaN (no sensitive columns in the gold set)
        return NAN
    if r == 0:
        return 0.0
    return 2 * p * r / (p + r) if p == p and (p + r) else NAN


def score_warnings(predicted, gold, basis, grounded=None, parse_ok=True):
    """
    Returns (metrics, detail).
      tp = sensitive columns flagged        fn = sensitive columns missed
      fp = non-sensitive columns flagged    (false alarms)
      hallucinated = flagged names that are not columns (counted apart from fp)

    grounded: True if the prompt asked for a ground per column (charter/gdpr prompts).
      Pass it explicitly (run_sweep and rescore do), so an EMPTY answer to a legal prompt
      still gets strict scores (strict recall 0) instead of being left out of the averages.
      None = guess from the answer (old behaviour: grounded if any flag carries a ground).
    parse_ok: False if the model's answer could not be read as JSON (parse_warnings' flag).
    """
    a1 = gold["activity_1"]
    by_name = {c["feature"]: c for c in a1["columns"]}
    all_cols = set(by_name)
    positive = {f for f, c in by_name.items() if c[basis]}

    flagged = {}
    for w in predicted:
        if w.get("activity") == ACTIVITY and w["feature"] not in flagged:
            flagged[w["feature"]] = w
    pred_raw = set(flagged)
    hallucinated = pred_raw - all_cols
    echoed = sorted(f for f in hallucinated if _norm(f) in GROUND_NAMES)
    echo = len(echoed) >= ECHO_MIN

    # An echo answer is graded as if it flagged no column (see the module docstring).
    pred = set() if echo else pred_raw

    tp = pred & positive
    fn = positive - pred
    fp = (pred & all_cols) - positive

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else NAN
    recall = len(tp) / len(positive) if positive else NAN

    metrics = {
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "tn": len(all_cols) - len(tp) - len(fp) - len(fn),
        "hallucinated": len(hallucinated),
        "precision": precision, "recall": recall, "f1": _f1(precision, recall),
        # false alarms normalised by the number of expected warnings, as in Ciatto's brief
        "false_alarm_rate": len(fp) / max(len(positive), 1),
        # the three kinds of failed answer (module docstring); at most one of them is 1
        "empty_answer": int(parse_ok and not pred_raw),
        "parse_failed": int(not parse_ok),
        "definition_echo": int(echo),
        "echoed_grounds": len(echoed),
    }

    # ---- error typing: false alarms per column family (all families logged, 0 if none)
    families = a1.get("families", {})
    fam_of = {f: by_name[f].get("family", "unlabelled") for f in fp}
    counts = Counter(fam_of.values())
    # fp_protected is only non-zero under the narrower basis: a column that is sensitive under
    # the Charter but not under GDPR Art. 9 (sex, age, ...) flagged under the GDPR key.
    for fam in sorted(set(families) | set(counts)):
        metrics[f"fp_{fam}"] = counts.get(fam, 0)
    metrics["fp_art10"] = sum(n for fam, n in counts.items()
                              if families.get(fam, {}).get("gdpr_art10", False))

    detail = {
        "hits": sorted(tp), "misses": sorted(fn),
        "false_alarms": sorted(fp), "hallucinated": sorted(hallucinated),
        "false_alarms_by_family": {fam: sorted(f for f in fp if fam_of[f] == fam) for fam in sorted(counts)},
        "grounds_given": {f: flagged[f].get("ground") for f in sorted(pred_raw)},
        "echoed_grounds": echoed,
    }

    # ---- strict scoring: only if the prompt asked for grounds
    if grounded is None:
        grounded = any(w.get("ground") for w in flagged.values())
    if grounded:
        key = f"{basis}_grounds"
        right, wrong = set(), {}
        for f in tp:
            accepted = {_norm(g) for g in by_name[f].get(key, [])}
            given = flagged[f].get("ground")
            if _norm(given) in accepted:
                right.add(f)
            else:
                wrong[f] = given
        in_data = len(tp) + len(fp)
        s_prec = len(right) / in_data if in_data else NAN
        s_rec = len(right) / len(positive) if positive else NAN
        metrics.update({
            "strict_tp": len(right),
            "wrong_ground": len(wrong),
            "strict_precision": s_prec,
            "strict_recall": s_rec,
            "strict_f1": _f1(s_prec, s_rec),
            # flags where the model itself said "none of the listed grounds"
            "off_definition": sum(1 for f in pred & all_cols
                                  if _norm(flagged[f].get("ground")) == _norm(OTHER_GROUND)),
            "missing_ground": sum(1 for f in pred & all_cols if not flagged[f].get("ground")),
        })
        detail["wrong_ground"] = dict(sorted(wrong.items()))

    return metrics, detail
