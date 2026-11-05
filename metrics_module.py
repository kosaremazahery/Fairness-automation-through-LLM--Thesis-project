import numpy as np 


def identify_sensitive_attributes_llm(df, llm):
    cols = list(df.columns)
    prompt = f"""
You are an expert in algorithmic fairness.

Here are the dataset columns:
{cols}

Task:
- Identify which of these are sensitive attributes. You MUST NOT mark outcome variables (like labels) or behaviour counts (e.g. priors, number of arrests) as sensitive attributes.
Focus on human characteristics and social categories only.
  (e.g., race, gender, age, nationality, religion, disability, income).
- Rank the attributes based on their importance. Use your domain knowledge.
- Return ONLY a comma-separated list of column names.
"""
    raw = llm(prompt, max_tokens=200)
    candidates = [c.strip() for c in raw.split(",") if c.strip() in cols]
    return candidates


def pick_main_sensitive_attribute(sensitive_attrs):
    return sensitive_attrs[0] if sensitive_attrs else None



def identify_groups_llm(df, sensitive_col, llm):
    counts = df[sensitive_col].value_counts(normalize=True).round(3)

    prompt = f"""
We are analyzing '{sensitive_col}'.

Here are the group frequencies:
{counts.to_string()}

Task:
- Choose ONE protected/disadvantaged group.
- Choose ONE reference/dominant group.
- Return EXACTLY:

protected_group: <group_name>
reference_group: <group_name>
"""
    raw = llm(prompt, max_tokens=200)

    protected, reference = None, None

    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("protected_group:"):
            protected = line.split(":", 1)[1].strip()
        if line.lower().startswith("reference_group:"):
            reference = line.split(":", 1)[1].strip()

    valid_groups = set(df[sensitive_col].dropna().unique())
    if protected not in valid_groups: protected = None
    if reference not in valid_groups: reference = None

    return protected, reference




def fairness_kpis_on_data(df, sensitive_col, group_a, group_b, target_col, positive_label=1):
    df_a = df[df[sensitive_col] == group_a]
    df_b = df[df[sensitive_col] == group_b]

    p_a = (df_a[target_col] == positive_label).mean()
    p_b = (df_b[target_col] == positive_label).mean()

    SPD = p_a - p_b
    DIR = p_a / p_b if p_b > 0 else np.nan

    return {
        "sensitive_col": sensitive_col,
        "group_a": group_a,
        "group_b": group_b,
        "p_a": float(p_a),
        "p_b": float(p_b),
        "SPD": float(SPD),
        "DIR": float(DIR),
    }



def llm_explain_and_mitigate(metrics, target_col, llm):
    prompt = f"""
You are an expert in algorithmic fairness. Only analyse in the database layer.
Keep in mind that you are analysing the true lables, not the model predictions.

Sensitive attribute: {metrics['sensitive_col']}
Protected group: {metrics['group_a']}
Reference group: {metrics['group_b']}

Outcome distribution (data-level):
P({target_col}=1 | protected) = {metrics['p_a']:.3f}
P({target_col}=1 | reference) = {metrics['p_b']:.3f}
SPD = {metrics['SPD']:.3f}
DIR = {metrics['DIR']:.3f}

Task:
1. Explain what these numbers mean.
2. Say whether they indicate potential unfairness.
3. Suggest reasonable computational fairness mitigation strategies in a clean list with no extra explanation(only name the method). Rank the methods based on the metrics and your knowledge.
Answer in clean paragraphs. 
"""
    return llm(prompt, max_tokens=700)



