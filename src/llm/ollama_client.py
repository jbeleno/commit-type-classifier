"""Thin Ollama client.

Talks to a local Ollama daemon (http://localhost:11434) via its REST API.
Returns the generated text plus latency in milliseconds. No streaming —
we collect the full response so we can measure end-to-end latency and
feed it into the evaluation harness.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120


@dataclass
class Generation:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OllamaError(RuntimeError):
    pass


def is_alive(host: str = OLLAMA_HOST) -> bool:
    try:
        r = requests.get(f"{host}/api/version", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(host: str = OLLAMA_HOST) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def chat(
    model: str,
    messages: list[Dict[str, Any]],
    *,
    tools: Optional[list[Dict[str, Any]]] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
    seed: Optional[int] = 42,
    fmt: Optional[str] = None,
    host: str = OLLAMA_HOST,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Conversational call to Ollama's /api/chat with optional tool definitions.

    Returns the raw ``message`` dict from Ollama, which may include
    ``tool_calls`` if the model decided to call one.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "seed": seed,
        },
    }
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt

    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise OllamaError(f"Ollama returned {r.status_code}: {r.text[:200]}")
    data = r.json()
    msg = data.get("message", {})
    msg["_latency_ms"] = elapsed
    msg["_prompt_tokens"] = int(data.get("prompt_eval_count", 0) or 0)
    msg["_completion_tokens"] = int(data.get("eval_count", 0) or 0)
    return msg


def generate(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 192,
    stop: Optional[list[str]] = None,
    seed: Optional[int] = 42,
    fmt: Optional[str] = None,
    host: str = OLLAMA_HOST,
    timeout: int = DEFAULT_TIMEOUT,
) -> Generation:
    """Single-shot generation. Returns full text + telemetry.

    fmt="json" forces strict JSON output (Ollama structured-output mode).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "seed": seed,
        },
    }
    if system:
        payload["system"] = system
    if stop:
        payload["options"]["stop"] = stop
    if fmt:
        payload["format"] = fmt

    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise OllamaError(f"Ollama returned {r.status_code}: {r.text[:200]}")
    data = r.json()
    return Generation(
        text=data.get("response", "").strip(),
        model=model,
        latency_ms=elapsed,
        prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
        completion_tokens=int(data.get("eval_count", 0) or 0),
        total_tokens=int(data.get("prompt_eval_count", 0) or 0) + int(data.get("eval_count", 0) or 0),
    )
