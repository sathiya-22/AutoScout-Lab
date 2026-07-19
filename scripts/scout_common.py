#!/usr/bin/env python3
"""Shared helpers for AutoScout scripts.

Problem-log access + a Gemini call wrapper with model fallback and retry.
google-genai is imported lazily so log utilities work without the SDK installed.
"""

import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
PROBLEM_LOG = REPO_ROOT / "problems" / "problem_log.jsonl"

# gemini-2.0-flash is the stable primary — 2.5-flash has chronic 503s on free tier.
PRIMARY_MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 2
RETRY_BASE_SEC = 30  # back-off: 30s, 60s


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_problems() -> list[dict]:
    if not PROBLEM_LOG.exists():
        return []
    problems = []
    for line in PROBLEM_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            problems.append(json.loads(line))
    return problems


def save_problems(problems: list[dict]) -> None:
    PROBLEM_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROBLEM_LOG.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in problems),
        encoding="utf-8",
    )


def parse_sections(raw: str) -> dict[str, str]:
    """Parse '=== path ===\\ncontent' sections into {path: content}.

    Paths come from model output, so absolute paths and '..' components are
    dropped — they would escape the temp checkout these files get written to."""
    files: dict[str, str] = {}
    pattern = r"=== ([\w./\-]+) ===\n(.*?)(?==== [\w./\-]+ ===|\Z)"
    for match in re.finditer(pattern, raw, re.DOTALL):
        path = match.group(1).strip()
        if path.startswith("/") or ".." in path.split("/"):
            print(f"  WARN: skipping unsafe path from model output: {path!r}",
                  file=sys.stderr)
            continue
        files[path] = match.group(2).strip()
    return files


def broken_python_files(files: dict[str, str]) -> list[str]:
    """Names of .py files in the set that fail to even compile — the
    cheapest possible 'did the model write valid code' check."""
    broken = []
    for path, content in files.items():
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as e:
                print(f"  WARN: {path} has a syntax error: {e}", file=sys.stderr)
                broken.append(path)
    return broken


def parse_json_lenient(raw: str):
    """Parse model JSON output: strip code fences; salvage truncated arrays
    by trimming to the last complete object."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if not raw.startswith("["):
            raise
        end = raw.rfind("}")
        while end != -1:
            try:
                return json.loads(raw[:end + 1] + "]")
            except json.JSONDecodeError:
                end = raw.rfind("}", 0, end)
        raise


def is_transient(err: str) -> bool:
    """True for errors worth retrying (rate limits, server overload).
    404 / auth errors are NOT transient — skip to next model immediately."""
    if any(x in err for x in ("404", "403", "401", "NOT_FOUND", "PERMISSION_DENIED")):
        return False
    keywords = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                "quota", "overloaded", "high demand", "retry")
    return any(k.lower() in err.lower() for k in keywords)


def call_gemini(api_key: str, prompt: str, system: str, max_tokens: int,
                json_mode: bool = False) -> str:
    """One prompt -> response text, trying primary then fallback model."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.4 if json_mode else 0.7,
            response_mime_type="application/json" if json_mode else None,
            # 2.5 models spend output tokens on thinking by default, which
            # truncates the visible response — disable it.
            thinking_config=(types.ThinkingConfig(thinking_budget=0)
                             if model.startswith("gemini-2.5") else None),
        )
        print(f"Trying model: {model}", flush=True)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config)
                if not response.text:
                    raise RuntimeError("empty response")
                print(f"  Success with {model} on attempt {attempt}.")
                return response.text
            except Exception as e:
                err = str(e)
                if is_transient(err) and attempt < MAX_RETRIES:
                    wait = RETRY_BASE_SEC * (2 ** (attempt - 1))
                    print(f"  Transient error (attempt {attempt}/{MAX_RETRIES}), "
                          f"retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  Moving on from {model} after {attempt} attempt(s): "
                          f"{err[:200]}", file=sys.stderr)
                    break  # try next model
    raise RuntimeError("all models and retries exhausted")
