# main.py

import pandas as pd
from llm_fairness import make_ollama_llm
from pipeline import fairness_analysis  # we'll create this

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def main():
    # 0) Load data
    df = load_dataset("compas-scores-two-years_v1.csv")

    # 1) Run your automated fairness assistantخ
    result = fairness_analysis(
        df=df,
        target_col="two_year_recid",
        llm=make_ollama_llm("llama3.1")  # this is the Ollama-based function
    )

    # 2) Print / save results
    print("Sensitive attributes suggested:", result["sensitive_attributes"])
    print("Chosen sensitive attribute:", result["chosen_sensitive_attribute"])
    print("Protected group:", result["protected_group"])
    print("Reference group:", result["reference_group"])
    print("\nMetrics:", result["metrics"])
    print("\nLLM explanation:\n")
    print(result["llm_explanation"])

if __name__ == "__main__":
    main()