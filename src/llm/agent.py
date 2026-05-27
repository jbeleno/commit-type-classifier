"""Agentic AI tab — conversational interface over the project's tooling.

Exposes the commit classifier, the LLM generator, and the pydriller-based
repository scanner as Ollama tools, then runs a chat loop where a local
LLM decides which tool to call and stitches the results back into the
conversation. This is the syllabus' Topic 11 ("Agentic AI") wired into
the project's existing infrastructure.

Tools available to the agent:

  classify_commit(message, diff, model)
      Predict the Conventional Commit type using any classifier or
      LLM mode in ``src.inference.AVAILABLE_MODELS``.

  generate_commit_message(diff, model)
      Run the hybrid LLM pipeline (RAG + LLM + classifier verifier) to
      author a commit message from a diff.

  scan_repo(path, last_n)
      Use pydriller to read the last N commits of a local git repo.
      Returns hash / author / date / message / diff per commit.

  classify_repo(path, last_n, model)
      Compose scan_repo + classify_commit and return the class histogram
      plus per-commit predictions.

  list_models() / list_classes()
      Discovery helpers so the agent can pick a valid argument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import TARGET_CLASSES
from src.inference import AVAILABLE_MODELS, predict
from src.llm import ollama_client
from src.llm.hybrid import hybrid_generate

DEFAULT_AGENT_MODEL = "llama3.2:3b-instruct-q4_K_M"
MAX_TURNS = 8
_TOOL_NAMES = {"classify_commit", "generate_commit_message", "scan_repo",
               "classify_repo", "list_models", "list_classes"}


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #


def _tool_classify_commit(message: str, diff: str = "", model: str = "baseline_tfidf") -> Dict[str, Any]:
    if model not in AVAILABLE_MODELS:
        return {"error": f"unknown model '{model}'. Use list_models()."}
    p = predict(message or "", diff or "", model)
    return {
        "label": p.label,
        "confidence": round(p.confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in p.probabilities.items()},
        "model": p.model,
    }


def _tool_generate_commit_message(diff: str, model: str = "qwen2.5-coder:3b") -> Dict[str, Any]:
    r = hybrid_generate(diff or "", model=model)
    return {
        "message": r.final_message,
        "type": r.final_type,
        "llm_type": r.llm_type,
        "verifier_type": r.verifier_type,
        "verifier_confidence": round(r.verifier_confidence, 4),
        "type_changed": r.type_changed,
        "latency_ms": round(r.latency_ms_total, 0),
        "retrieved_examples": [
            {"type": e.type, "subject": e.subject[:80], "score": round(e.score, 4)}
            for e in r.retrieved
        ],
    }


def _tool_scan_repo(path: str, last_n: int = 20) -> Dict[str, Any]:
    try:
        from pydriller import Repository
    except ImportError as exc:  # noqa: BLE001
        return {"error": f"pydriller required: {exc}"}

    try:
        last_n = int(last_n)
    except (TypeError, ValueError):
        last_n = 20
    last_n = max(1, min(last_n, 200))

    p = Path(path).expanduser().resolve()
    if not (p / ".git").exists():
        return {"error": f"not a git repository: {p}"}

    commits: List[Dict[str, Any]] = []
    for i, commit in enumerate(
        Repository(str(p), only_no_merge=True, order="reverse").traverse_commits()
    ):
        if i >= last_n:
            break
        diff_text = "\n".join(
            f"diff --git a/{m.old_path or m.new_path} b/{m.new_path or m.old_path}\n{m.diff or ''}"
            for m in commit.modified_files
        )
        commits.append({
            "hash": commit.hash[:10],
            "author": commit.author.name,
            "date": commit.committer_date.isoformat(),
            "message": commit.msg.splitlines()[0][:160] if commit.msg else "",
            "diff_preview": diff_text[:500],
            "_full_diff": diff_text,
        })
    return {"path": str(p), "n_commits": len(commits), "commits": commits}


def _tool_classify_repo(path: str, last_n: int = 20, model: str = "baseline_tfidf") -> Dict[str, Any]:
    try:
        last_n = int(last_n)
    except (TypeError, ValueError):
        last_n = 20
    if model not in AVAILABLE_MODELS:
        return {"error": f"unknown model '{model}'. Use list_models()."}
    scan = _tool_scan_repo(path, last_n)
    if "error" in scan:
        return scan
    histogram: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    per_commit: List[Dict[str, Any]] = []
    for c in scan["commits"]:
        p = predict(c["message"], c["_full_diff"], model)
        histogram[p.label] += 1
        per_commit.append({
            "hash": c["hash"],
            "message": c["message"][:80],
            "predicted_type": p.label,
            "confidence": round(p.confidence, 3),
        })
    total = sum(histogram.values()) or 1
    distribution = {k: f"{(v / total) * 100:.1f}%" for k, v in histogram.items()}
    return {
        "path": scan["path"],
        "model": model,
        "n_commits": scan["n_commits"],
        "histogram": histogram,
        "distribution_pct": distribution,
        "per_commit": per_commit,
    }


def _tool_list_models() -> Dict[str, Any]:
    return {"models": AVAILABLE_MODELS, "default_classifier": "baseline_tfidf",
            "default_llm": "qwen2.5-coder:3b"}


def _tool_list_classes() -> Dict[str, Any]:
    return {"classes": list(TARGET_CLASSES)}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "classify_commit": _tool_classify_commit,
    "generate_commit_message": _tool_generate_commit_message,
    "scan_repo": _tool_scan_repo,
    "classify_repo": _tool_classify_repo,
    "list_models": _tool_list_models,
    "list_classes": _tool_list_classes,
}


# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI-style, accepted by Ollama's /api/chat)
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_commit",
            "description": "Predict the Conventional Commit type for a single commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The commit message."},
                    "diff": {"type": "string", "description": "The unified diff. Optional."},
                    "model": {"type": "string",
                              "description": "One of: baseline_tfidf, cnn_text, distilbert, codebert, ensemble, llm:qwen2.5-coder:3b, llm-ensemble. Default baseline_tfidf."},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_commit_message",
            "description": "Author a Conventional Commit message from a diff using the hybrid LLM pipeline (RAG + LLM + verifier).",
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {"type": "string", "description": "The unified diff text."},
                    "model": {"type": "string",
                              "description": "Ollama model tag, default qwen2.5-coder:3b."},
                },
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_repo",
            "description": "Read the last N commits from a local Git repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Absolute or ~-relative path to a local git repo."},
                    "last_n": {"type": "integer",
                               "description": "How many recent commits to read. Default 20."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_repo",
            "description": "Scan + classify the last N commits of a local git repo and return a class histogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a local git repo."},
                    "last_n": {"type": "integer", "description": "Default 20."},
                    "model": {"type": "string",
                              "description": "Classifier or LLM model. Default baseline_tfidf."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List the available classifier and LLM model names.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_classes",
            "description": "List the five Conventional Commit target classes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


SYSTEM_PROMPT = (
    "You are an assistant embedded in a commit-classification project. "
    "You have access to tools that classify a single commit, generate a "
    "commit message from a diff, scan a local git repository, and run "
    "a full classify-the-last-N-commits sweep over a repo. "
    "When the user asks about a repository, prefer `classify_repo`. "
    "When the user pastes a diff and asks for a message, call "
    "`generate_commit_message`. When the user pastes a commit message + "
    "diff and asks for a label, call `classify_commit`. "
    "If the user did not specify a model, use the default. "
    "AFTER A TOOL RETURNS: the GUI already renders the result as a table "
    "or chart, so DO NOT repeat the data line by line. Write 1-3 SHORT "
    "sentences that INTERPRET the result: what stands out, what is "
    "surprising, what the user should look at, or an actionable next "
    "step. Be specific to the numbers you saw. Never paste JSON, never "
    "list every commit, never restate percentages already shown."
)


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


@dataclass
class AgentStep:
    role: str                            # "user" | "assistant" | "tool"
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


def _parse_tool_call_from_text(text: str) -> Optional[Dict[str, Any]]:
    """When a small model emits the tool call as a JSON literal in `content`
    instead of the proper `tool_calls` field, recover it."""
    import re

    # strip code fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if name in _TOOL_NAMES:
        return {"function": {"name": name, "arguments": args}}
    return None


def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{name} failed: {exc.__class__.__name__}: {exc}"}


def _truncate_tool_result(result: Dict[str, Any], limit: int = 4000) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + '..."<truncated>"'


def run_agent(
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    *,
    model: str = DEFAULT_AGENT_MODEL,
    max_turns: int = MAX_TURNS,
) -> tuple[List[AgentStep], List[Dict[str, Any]]]:
    """Run a conversation turn that may involve multiple tool calls.

    Returns (visible_steps, full_message_history). ``visible_steps`` is
    what the GUI should render in order; ``full_message_history`` is what
    to pass back as ``history`` on the next user turn.
    """
    messages: List[Dict[str, Any]] = list(history or [])
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": user_message})
    steps: List[AgentStep] = [AgentStep(role="user", content=user_message)]

    for _ in range(max_turns):
        msg = ollama_client.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
        latency = float(msg.get("_latency_ms", 0))
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content", "") or ""

        # Fallback: some small models emit the tool call as JSON in `content`
        # instead of populating `tool_calls`. Try to parse it.
        if not tool_calls and content:
            parsed = _parse_tool_call_from_text(content)
            if parsed is not None:
                tool_calls = [parsed]
                content = ""

        if not tool_calls:
            steps.append(AgentStep(role="assistant", content=content, latency_ms=latency))
            messages.append({"role": "assistant", "content": content})
            break

        # The model decided to call one or more tools.
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            fn_block = call.get("function", {}) or {}
            name = fn_block.get("name", "")
            raw_args = fn_block.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}

            result = _execute_tool(name, args)
            steps.append(AgentStep(
                role="tool",
                tool_name=name,
                tool_args=args,
                tool_result=result,
                latency_ms=latency,
            ))
            messages.append({
                "role": "tool",
                "content": _truncate_tool_result(result),
                "name": name,
            })
            latency = 0.0  # only attribute the LLM latency once per turn
    else:
        steps.append(AgentStep(
            role="assistant",
            content="(agent reached max-turn limit without producing a final answer)",
        ))
        messages.append({"role": "assistant", "content": steps[-1].content})

    return steps, messages
