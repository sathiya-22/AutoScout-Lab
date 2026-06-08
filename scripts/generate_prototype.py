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
MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 5000
MAX_RETRIES = 3
RETRY_WAIT_SEC = 65  # just over 60s to clear the per-minute window


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


def generate_prototype_files(client: genai.Client, topic: str) -> dict[str, str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=USER_TEMPLATE.format(topic=topic),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.7,
                ),
            )
            raw = response.text
            files = parse_sections(raw)
            if not files:
                print("ERROR: no sections found in model response.", file=sys.stderr)
                print(raw[:1000], file=sys.stderr)
                sys.exit(1)
            return files

        except Exception as e:
            err = str(e)
            is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()
            if is_quota and attempt < MAX_RETRIES:
                print(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {RETRY_WAIT_SEC}s before retry...", flush=True)
                time.sleep(RETRY_WAIT_SEC)
            else:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)

    print("ERROR: all retries exhausted.", file=sys.stderr)
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
    print(f"Model        : {MODEL}")
    print(f"Max output   : {MAX_OUTPUT_TOKENS} tokens  "
          f"(budget used: ~{MAX_OUTPUT_TOKENS/250000*100:.1f}% of 250k daily limit)")

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
