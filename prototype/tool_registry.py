from tools import (
    tool_list_columns,
    tool_value_counts,
    tool_compute_label_fairness,
)

TOOL_REGISTRY = {
    "list_columns": tool_list_columns,
    "value_counts": tool_value_counts,
    "compute_label_fairness": tool_compute_label_fairness,
    # "finish" will be handled specially in the agent loop
}