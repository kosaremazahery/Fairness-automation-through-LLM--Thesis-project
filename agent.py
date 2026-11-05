
import json
from typing import Any, Dict, List, Callable
from tool_registry import TOOL_REGISTRY
from agent_prompt import BASE_AGENT_PROMPT

def parse_json_safely(raw: str) -> Dict[str, Any]:
    """Try to parse JSON, strip code fences if present."""
    try:
        return json.loads(raw)
    except Exception:
        cleaned = (
            raw.strip()
               .replace("```json", "")
               .replace("```", "")
        )
        return json.loads(cleaned)

def build_agent_prompt(history: List[Dict[str, Any]]) -> str:
    """
    Build the full prompt: base instructions + previous calls/results.
    """
    if not history:
        return BASE_AGENT_PROMPT + "\n\nNow choose the first function to call."

    history_text = "Here is the history of function calls and their results so far:\n"
    for step, h in enumerate(history):
        history_text += f"\nSTEP {step+1}:\n"
        history_text += f"Called: {h['function']} with args: {h['args']}\n"
        history_text += f"Result: {h['result']}\n"

    return BASE_AGENT_PROMPT + "\n\n" + history_text + "\n\nNow choose the NEXT function to call."

def run_fairness_agent(
    df,
    llm: Callable[[str, int], str],
    max_steps: int = 10,
) -> Dict[str, Any]:
    """
    Main loop: let the LLM decide which tools to use.
    Stops after 'finish' or max_steps.
    """
    history: List[Dict[str, Any]] = []
    final_summary = None

    for step in range(max_steps):
        prompt = build_agent_prompt(history)
        raw_response = llm(prompt, max_tokens=600)

        try:
            instruction = parse_json_safely(raw_response)
        except Exception:
            print("Failed to parse LLM JSON:\n", raw_response)
            break

        func_name = instruction.get("function")
        args = instruction.get("args", {}) or {}

        if func_name == "finish":
            final_summary = args.get("summary", "")
            print("\n[AGENT] Decided to finish.")
            print("Summary:\n", final_summary)
            break

        if func_name not in TOOL_REGISTRY:
            print(f"Unknown function requested by LLM: {func_name}")
            break

        tool_fn = TOOL_REGISTRY[func_name]
        try:
            result = tool_fn(df, **args)
        except TypeError as e:
            result = {"error": f"Bad arguments for {func_name}: {e}"}

        step_record = {"function": func_name, "args": args, "result": result}
        history.append(step_record)

        print(f"\n[STEP {step+1}] Called {func_name} with args={args}")
        print("Result:", result)

    return {"history": history, "final_summary": final_summary}
