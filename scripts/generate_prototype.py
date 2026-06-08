#!/usr/bin/env python3
"""AutoScout daily prototype generator.

Runs inside GitHub Actions every day. Picks a fresh AI/ML topic, calls
Gemini 2.5 Flash (1M input / 65k output tokens, free-tier available) to
generate a complete prototype project, then writes it into the dated batch
folder so the workflow can commit and push it.
"""

import os
import re
import sys
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

# ── Topic pool ──────────────────────────────────────────────────────────────
# Add more entries here as the repo grows. The picker avoids repeats by
# checking which slugs already exist in the repo.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def pick_topic(repo_root: Path) -> str:
    """Return a topic not already present in the repo, cycling by day-of-year."""
    used_slugs: set[str] = set()
    for p in repo_root.glob("ai_scout_batch_*/*/"):
        used_slugs.add(p.name)

    day_index = date.today().timetuple().tm_yday
    for i in range(len(TOPICS)):
        candidate = TOPICS[(day_index + i) % len(TOPICS)]
        if slugify(candidate) + "-prototype" not in used_slugs:
            return candidate

    # All topics used — fall back to day index (will repeat eventually)
    return TOPICS[day_index % len(TOPICS)]


def parse_sections(raw: str) -> dict[str, str]:
    """Extract === FILENAME === sections from the model's response."""
    files: dict[str, str] = {}
    pattern = r"=== ([\w./\-]+) ===\n(.*?)(?==== [\w./\-]+ ===|\Z)"
    for match in re.finditer(pattern, raw, re.DOTALL):
        name = match.group(1).strip()
        content = match.group(2).strip()
        files[name] = content
    return files


# ── Generation ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are AutoScout, an automated AI prototype generator. You produce
production-quality, self-contained Python prototype projects. Every project
must be realistic, well-structured, and immediately runnable (given an API key).
"""

USER_TEMPLATE = """\
Generate a complete Python prototype project for the topic:

  **{topic}**

Respond with exactly 4 sections using these exact headers (no markdown fences
inside the sections):

=== README.md ===
=== main.py ===
=== config.py ===
=== requirements.txt ===

Guidelines:
- README.md  : 250–400 words. Problem statement, approach, key components, usage.
- main.py    : 150–300 lines. Functional demo of the prototype. Use the
               google-genai SDK (model: gemini-2.5-flash) as the LLM provider.
               Load the API key from the GEMINI_API_KEY env var.
- config.py  : Pydantic BaseSettings class with fields: model_name, temperature,
               max_tokens, api_key (from env). Include a ValidationConfig or
               similar nested config relevant to the topic.
- requirements.txt: Only packages actually imported in main.py / config.py,
               one per line, no version pins.

Do NOT wrap file contents in markdown code fences.
"""

# gemini-2.5-flash: 1M input tokens, 65,536 output tokens — largest free-tier window
MODEL = "gemini-2.5-flash"


def generate_prototype_files(client: genai.Client, topic: str) -> dict[str, str]:
    response = client.models.generate_content(
        model=MODEL,
        contents=USER_TEMPLATE.format(topic=topic),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=65536,
            temperature=0.7,
        ),
    )
    raw = response.text
    files = parse_sections(raw)
    if not files:
        print("ERROR: Could not parse any sections from model response.", file=sys.stderr)
        print("--- raw response ---", file=sys.stderr)
        print(raw[:2000], file=sys.stderr)
        sys.exit(1)
    return files


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
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

    print(f"Topic   : {topic}")
    print(f"Output  : {batch_name}/{proto_slug}/")
    print(f"Model   : {MODEL}  (1M input / 65k output tokens)")

    files = generate_prototype_files(client, topic)

    proto_dir.mkdir(parents=True, exist_ok=True)

    # Write generated files
    for filename, content in files.items():
        target = proto_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
        print(f"  wrote {filename}  ({len(content):,} chars)")

    # Ensure a src package stub exists (matches existing repo convention)
    src_init = proto_dir / "src" / "__init__.py"
    src_init.parent.mkdir(exist_ok=True)
    if not src_init.exists():
        src_init.touch()

    print(f"\nDone. Prototype written to {proto_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
