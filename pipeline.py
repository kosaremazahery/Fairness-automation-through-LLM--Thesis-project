from typing import Dict
from metrics_module import (
    identify_sensitive_attributes_llm,
    pick_main_sensitive_attribute,
    identify_groups_llm,
    fairness_kpis_on_data,  # or _on_data
    llm_explain_and_mitigate,
)

def fairness_analysis(df, target_col: str, llm) -> Dict:
    sensitive_attrs = identify_sensitive_attributes_llm(df, llm)
    sensitive_col = pick_main_sensitive_attribute(sensitive_attrs)

    prot_group, ref_group = identify_groups_llm(df, sensitive_col, llm)

    # choose whether you want data-level or model-level metrics here
    metrics = fairness_kpis_on_data(
        df=df,
        sensitive_col=sensitive_col,
        group_a=prot_group,
        group_b=ref_group,
        target_col=target_col,
          
    )

    explanation = llm_explain_and_mitigate(metrics, target_col, llm)

    return {
        "sensitive_attributes": sensitive_attrs,
        "chosen_sensitive_attribute": sensitive_col,
        "protected_group": prot_group,
        "reference_group": ref_group,
        "metrics": metrics,
        "llm_explanation": explanation,
    }