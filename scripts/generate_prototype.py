#!/usr/bin/env python3
"""AutoScout daily prototype generator.

Free-tier budget: 5 RPM, 20 RPD, 250k tokens/day.
Strategy: 1 request per run, ~5k output tokens — uses <2% of daily budget.
Retries up to 3 times with 65-second back-off on rate-limit errors.
"""

import os
import re
import sys
import time
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

# 1 request per day, ~5k output tokens ≈ <2% of the 250k daily token budget
# Fallback is tried if primary exhausts all retries due to server-side demand.
PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.0-flash"   # stable v1beta model, not subject to 2.5 demand spikes
MAX_OUTPUT_TOKENS = 5000
MAX_RETRIES = 5
RETRY_BASE_SEC = 30  # exponential back-off: 30, 60, 120, 240, 480s


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def pick_topic(repo_root: Path) -> str:
    used_slugs: set[str] = set()
    for p in repo_root.glob("ai_scout_batch_*/*/"):
        used_slugs.add(p.name)

    day_index = date.today().timetuple().tm_yday
    for i in range(len(TOPICS)):
        candidate = TOPICS[(day_index + i) % len(TOPICS)]
        if slugify(candidate) + "-prototype" not in used_slugs:
            return candidate

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    repo_root = Path(__file__).parent.parent.resolve()
    today = date.today()
    batch_name = f"ai_scout_batch_{today.strftime('%Y_%m_%d')}"
    batch_dir = repo_root / batch_name

    if batch_dir.exists():
        print(f"Batch {batch_name} already exists — nothing to do.")
        sys.exit(0)

    topic = pick_topic(repo_root)
    proto_slug = slugify(topic) + "-prototype"
    proto_dir = batch_dir / proto_slug

    print(f"Topic        : {topic}")
    print(f"Output       : {batch_name}/{proto_slug}/")
    print(f"Primary      : {PRIMARY_MODEL}  →  fallback: {FALLBACK_MODEL}")
    print(f"Max output   : {MAX_OUTPUT_TOKENS} tokens  "
          f"(~{MAX_OUTPUT_TOKENS/250000*100:.1f}% of 250k daily limit)")

    files = generate_prototype_files(client, topic)

    proto_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        target = proto_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        print(f"  wrote {filename}  ({len(content):,} chars)")

    (proto_dir / "src").mkdir(exist_ok=True)
    (proto_dir / "src" / "__init__.py").touch()

    print(f"\nDone — {proto_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
