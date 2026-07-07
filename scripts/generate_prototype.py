#!/usr/bin/env python3
"""AutoScout daily prototype generator.

Each run generates one AI prototype and pushes it to a brand-new, standalone
public GitHub repo named "<topic-slug>-<date>" (via SCOUT_PAT) rather than
committing it into this repo — this repo only orchestrates generation.

Free-tier budget: 5 RPM, 20 RPD, 250k tokens/day.
Strategy: 1 request per run, ~5k output tokens — uses <2% of daily budget.
Primary model: gemini-2.0-flash (stable). Falls back to gemini-2.5-flash.
Max 2 retries per model to preserve daily request quota.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

# ── Topic pool ───────────────────────────────────────────────────────────────
TOPICS = [
    "Adaptive RAG with Query Complexity Routing",
    "Self-Healing Agent Loop with Error Recovery",
    "Multi-Modal Embedding Fusion for Retrieval",
    "Constitutional AI Filter Pipeline",
    "LLM-Driven Data Augmentation Framework",
    "Streaming Token Budget Manager",
    "Hierarchical Memory Agent",
    "Semantic Deduplication at Ingestion",
    "LLM Uncertainty Quantification",
    "Tool Use Chain-of-Thought Verifier",
    "Cross-Lingual RAG Adapter",
    "Agent Observability and Trace Logger",
    "Prompt Compression via Selective Context",
    "Structured Output Validator with Auto-Retry",
    "Sparse-Dense Hybrid Search Ranker",
    "Retrieval Augmented Code Generation",
    "Persona-Conditioned Response Generator",
    "Dynamic Few-Shot Example Selector",
    "Knowledge Graph Augmented LLM",
    "Incremental Document Indexing Pipeline",
    "LLM Cost Optimizer with Tier Routing",
    "Adversarial Prompt Detection Filter",
    "Agentic Web Scraper with Extraction LLM",
    "Temporal Reasoning Module for QA",
    "Feedback-Driven Prompt Evolution System",
    "Contrastive Retrieval Reranker",
    "Hallucination Detection via Consistency Checks",
    "Long-Context Summarization with Rolling Window",
    "Multi-Hop Question Answering Agent",
    "Embedding Cache with Approximate Nearest Neighbour",
    "Retrieval Confidence Calibration Module",
    "Agentic Code Review Assistant",
    "Task Decomposition and Planning Agent",
    "LLM-Powered Data Cleaning Pipeline",
    "Semantic Router for Multi-Agent Dispatch",
]

# gemini-2.0-flash is the stable primary — 2.5-flash has chronic 503s on free tier.
# 2 retries max per model: keeps total requests to ≤6 worst-case (well under 20 RPD).
PRIMARY_MODEL  = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 5000
MAX_RETRIES = 2
RETRY_BASE_SEC = 30  # back-off: 30s, 60s

GITHUB_API = "https://api.github.com"


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def pick_topic() -> str:
    day_index = date.today().timetuple().tm_yday
    return TOPICS[day_index % len(TOPICS)]


def parse_sections(raw: str) -> dict[str, str]:
    files: dict[str, str] = {}
    pattern = r"=== ([\w./\-]+) ===\n(.*?)(?==== [\w./\-]+ ===|\Z)"
    for match in re.finditer(pattern, raw, re.DOTALL):
        files[match.group(1).strip()] = match.group(2).strip()
    return files


# ── Generation ───────────────────────────────────────────────────────────────

# Kept deliberately short to minimise input tokens (every token counts).
SYSTEM_PROMPT = (
    "You are AutoScout, an automated AI prototype generator. "
    "Output production-quality, self-contained Python projects. "
    "Be concise — prioritise clarity and correctness over verbosity."
)

USER_TEMPLATE = """\
Generate a Python prototype project for: {topic}

Reply with exactly these 4 section headers (no markdown fences inside):

=== README.md ===
=== main.py ===
=== config.py ===
=== requirements.txt ===

Rules:
- README.md  : 150-200 words. Problem, approach, usage.
- main.py    : 80-120 lines. Working demo using google-genai SDK \
(model: gemini-2.5-flash). Read GEMINI_API_KEY from env.
- config.py  : Pydantic BaseSettings with model_name, temperature, \
max_tokens, api_key fields.
- requirements.txt: only packages actually used, one per line, no versions.
"""


def _call_model(client: genai.Client, model: str, topic: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=USER_TEMPLATE.format(topic=topic),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
        ),
    )
    return response.text


def _is_transient(err: str) -> bool:
    """True for errors worth retrying (rate limits, server overload).
    404 / auth errors are NOT transient — skip to next model immediately."""
    if any(x in err for x in ("404", "403", "401", "NOT_FOUND", "PERMISSION_DENIED")):
        return False
    keywords = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                "quota", "overloaded", "high demand", "retry")
    return any(k.lower() in err.lower() for k in keywords)


def generate_prototype_files(client: genai.Client, topic: str) -> dict[str, str]:
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        print(f"Trying model: {model}", flush=True)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = _call_model(client, model, topic)
                files = parse_sections(raw)
                if not files:
                    print("ERROR: no sections found in model response.", file=sys.stderr)
                    print(raw[:1000], file=sys.stderr)
                    sys.exit(1)
                print(f"  Success with {model} on attempt {attempt}.")
                return files

            except Exception as e:
                err = str(e)
                if _is_transient(err) and attempt < MAX_RETRIES:
                    wait = RETRY_BASE_SEC * (2 ** (attempt - 1))  # 30,60,120,240s
                    print(f"  Transient error (attempt {attempt}/{MAX_RETRIES}), "
                          f"retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  Moving on from {model} after {attempt} attempt(s): {err[:200]}",
                          file=sys.stderr)
                    break  # try next model

    print("ERROR: all models and retries exhausted.", file=sys.stderr)
    sys.exit(1)


# ── GitHub repo creation ─────────────────────────────────────────────────────

def _gh_api(method: str, path: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{GITHUB_API}{path}", data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise RuntimeError(f"GitHub API {method} {path} failed: {e.code} {e.read().decode()[:300]}") from e


def get_authenticated_user(token: str) -> str:
    return _gh_api("GET", "/user", token)["login"]


def repo_exists(owner: str, name: str, token: str) -> bool:
    return bool(_gh_api("GET", f"/repos/{owner}/{name}", token))


def create_repo(name: str, token: str) -> dict:
    return _gh_api("POST", "/user/repos", token, {
        "name": name,
        "private": False,
        "description": "Auto-generated AI prototype by AutoScout",
        "auto_init": False,
    })


def push_prototype(clone_url: str, files: dict[str, str], token: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proto_dir = Path(tmp)
        for filename, content in files.items():
            target = proto_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
            print(f"  wrote {filename}  ({len(content):,} chars)")

        (proto_dir / "src").mkdir(exist_ok=True)
        (proto_dir / "src" / "__init__.py").touch()

        authed_url = clone_url.replace("https://", f"https://x-access-token:{token}@")

        def run(*args: str) -> None:
            subprocess.run(args, cwd=proto_dir, check=True)

        run("git", "init", "-q")
        run("git", "config", "user.name", "AutoScout Bot")
        run("git", "config", "user.email", "autoscout-bot@users.noreply.github.com")
        run("git", "add", ".")
        run("git", "commit", "-q", "-m", "feat: initial AutoScout prototype")
        run("git", "branch", "-M", "main")
        run("git", "remote", "add", "origin", authed_url)
        run("git", "push", "-q", "-u", "origin", "main")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    gh_token = os.environ.get("SCOUT_PAT", "")
    if not gh_token:
        print("ERROR: SCOUT_PAT is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=gemini_key)

    today = date.today()
    topic = pick_topic()
    repo_name = f"{slugify(topic)}-{today.isoformat()}"

    owner = get_authenticated_user(gh_token)

    if repo_exists(owner, repo_name, gh_token):
        print(f"Repo {owner}/{repo_name} already exists — nothing to do.")
        sys.exit(0)

    print(f"─── AutoScout: generating 1 project ───")
    print(f"Topic      : {topic}")
    print(f"New repo   : {owner}/{repo_name}")
    print(f"Models     : {PRIMARY_MODEL}  →  fallback: {FALLBACK_MODEL}")
    print(f"Tokens     : up to {MAX_OUTPUT_TOKENS} output  "
          f"(~{MAX_OUTPUT_TOKENS/250000*100:.1f}% of 250k daily budget)")

    files = generate_prototype_files(client, topic)

    repo = create_repo(repo_name, gh_token)
    push_prototype(repo["clone_url"], files, gh_token)

    print(f"\nDone — https://github.com/{owner}/{repo_name}")


if __name__ == "__main__":
    main()
