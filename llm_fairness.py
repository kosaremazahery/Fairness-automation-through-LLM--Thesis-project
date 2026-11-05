from typing import Callable, Optional
import ollama


def make_ollama_llm(
    model_name: str = "llama3.2:1b",
    temperature: float = 0.0,
    seed: int = 0,
    num_ctx: int = 8192,
    call_log: Optional[list] = None,
) -> Callable[..., str]:
    """
    Returns llm(prompt, max_tokens=800, schema=None) -> str.

    - temperature / seed : fixed for reproducibility
    - num_ctx            : Ollama's default context can be too short for a
                           53-column prompt; 8192 is safe for these models
    - schema             : optional JSON schema; Ollama then constrains the
                           output to valid JSON of that shape
    - call_log           : if a list is given, every prompt/response is
                           appended to it so failed runs can be inspected
    """
    def _llm(prompt: str, max_tokens: int = 800, schema: Optional[dict] = None) -> str:
        kwargs = {"format": schema} if schema is not None else {}
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
            **kwargs,
        )
        text = resp["message"]["content"]
        if call_log is not None:
            call_log.append({"prompt": prompt, "response": text, "used_schema": schema is not None})
        return text

    return _llm


def runtime_info(model_name: str) -> dict:
    """
    Which Ollama version and which exact build of the model a run used.

    Why: the same model name (e.g. 'llama3.2:3b') can point to a different build after
    `ollama pull`, and the Ollama app on Windows updates itself. Either can change answers
    even with the same prompt, temperature and seed, so both are recorded with every run.
    Returns "unknown" for anything it cannot read; never raises.
    """
    import json
    import os
    import urllib.request

    info = {"ollama_version": "unknown", "model_digest": "unknown"}
    try:
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("0.0.0.0", "127.0.0.1")
        if not host.startswith("http"):
            host = "http://" + host
        if host.count(":") == 1:                       # no port given
            host += ":11434"
        # no proxy: a VPN or system proxy must not intercept a call to this computer
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(host.rstrip("/") + "/api/version", timeout=5) as r:
            info["ollama_version"] = json.load(r).get("version", "unknown")
    except Exception:
        pass
    try:
        listed = ollama.list()
        models = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
        for m in models:
            if isinstance(m, dict):
                name, digest = m.get("model") or m.get("name"), m.get("digest")
            else:
                name, digest = getattr(m, "model", None) or getattr(m, "name", None), getattr(m, "digest", None)
            if name == model_name and digest:
                info["model_digest"] = str(digest)[:12]
    except Exception:
        pass
    return info
