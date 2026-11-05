
import pandas as pd
import numpy as np
from metrics_module import fairness_kpis_on_data

def tool_list_columns(df: pd.DataFrame):
    """Return all column names."""
    return {"columns": df.columns.tolist()}

def tool_value_counts(df: pd.DataFrame, column: str, top_n: int = 10):
    """Return value counts for a given column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    vc = df[column].value_counts().head(top_n).to_dict()
    return {"column": column, "value_counts": vc}

def tool_compute_label_fairness(
    df: pd.DataFrame,
    sensitive_col: str,
    group_a: str,
    group_b: str,
    target_col: str,
    positive_label: int = 1,
):
    """
    Wrapper around fairness_kpis_on_data:
    SPD/DIR on the *true labels* between two groups.
    """
    if sensitive_col not in df.columns or target_col not in df.columns:
        return {"error": "Invalid column name."}

    metrics = fairness_kpis_on_data(
        df=df,
        sensitive_col=sensitive_col,
        group_a=group_a,
        group_b=group_b,
        target_col=target_col,
        positive_label=positive_label,
    )

    return metrics
