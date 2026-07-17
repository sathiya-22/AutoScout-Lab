#!/usr/bin/env python3
"""AutoScout daily prototype generator.

Each run generates one prototype in the agentic-AI space — agent
orchestration, memory, evaluation/observability, guardrails/security,
multi-agent coordination, tool-calling/MCP protocols, agent ops/deployment —
and pushes it to a brand-new, standalone public GitHub repo named
"<topic-slug>-<date>" (via SCOUT_PAT) rather than committing it into this
repo — this repo only orchestrates generation.

The model picks whatever project format/stack best fits the problem (Python
package, Node/TS CLI, Go service, MCP server, browser extension, spec +
reference impl, ...) instead of a fixed Python-only shape, and only wires in
GEMINI_API_KEY when the idea's core feature genuinely needs an LLM call.

Topic selection: the highest-signal unaddressed problem from
problems/problem_log.jsonl (populated by scout_problems.py); the static
TOPICS pool is only a fallback for when the log has nothing new.

Free-tier budget: 5 RPM, 20 RPD, 250k tokens/day.
Strategy: 1 request per run, ~7k output tokens — uses <3% of daily budget.
Primary model: gemini-2.0-flash (stable). Falls back to gemini-2.5-flash.
Max 2 retries per model to preserve daily request quota.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from digest import update_section
from repo_registry import add_repo
from scout_common import call_gemini, load_problems, parse_sections, save_problems

# ── Fallback topic pool (used only when the problem log has nothing new) ─────
# Themed around agent orchestration, memory, evals/observability,
# guardrails/security, multi-agent coordination, tool-calling/MCP, and ops.
TOPICS = [
    "Agent Orchestration Graph with Conditional Branching",
    "Self-Healing Agent Loop with Error Recovery",
    "Hierarchical Agent Memory with Decay and Recall",
    "Agent Trace Logger and Replay Debugger",
    "Multi-Agent Task Handoff Protocol",
    "MCP Server Boilerplate with Auto-Discovered Tools",
    "Agent Guardrail Middleware for Unsafe Tool Calls",
    "Tool Use Chain-of-Thought Verifier",
    "Agent Cost and Token Budget Governor",
    "Structured Output Validator with Auto-Retry",
    "Agent-to-Agent Message Bus with Delivery Guarantees",
    "Sandboxed Tool Execution Environment for Agents",
    "Agent Evaluation Harness with Regression Scoring",
    "Human-in-the-Loop Approval Gate for Agent Actions",
    "Agent State Checkpointing and Rollback",
    "Multi-Agent Debate and Consensus Protocol",
    "Rate-Limited Tool Dispatcher for Agent Fleets",
    "Agent Permission Scoping and Capability Tokens",
    "Long-Running Agent Task Queue with Resumability",
    "Agent Observability Dashboard for Tool-Call Traces",
    "Dynamic Tool Discovery and Registration for Agents",
    "Agent Prompt Injection Detection Filter",
    "Multi-Agent Role Assignment and Load Balancer",
    "Agent Memory Consolidation via Summarization",
    "Local-First Agent Runtime for Resource-Constrained Hosts",
    "Agent Config Version Control and Diffing",
    "Streaming Tool-Call Parser with Partial Execution",
    "Agent Failure Triage and Auto-Retry Policy Engine",
    "Cross-Framework Agent Adapter (LangGraph/CrewAI/AutoGen)",
    "Agent Sandbox Escape Detection",
    "Task Decomposition and Planning Agent",
    "Agent Fleet Health Monitor with Alerting",
    "Semantic Router for Multi-Agent Dispatch",
    "Agent Skill Packaging and Sharing Format",
]

MAX_OUTPUT_TOKENS = 7000  # varied project shapes need more room than a fixed 4-file layout

GITHUB_API = "https://api.github.com"


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def pick_topic() -> str:
    day_index = date.today().timetuple().tm_yday
    return TOPICS[day_index % len(TOPICS)]


def pick_problem() -> dict | None:
    """Highest-signal problem from the scouted log not yet prototyped."""
    fresh = [p for p in load_problems() if p.get("status") == "new"]
    if not fresh:
        return None
    fresh.sort(key=lambda p: (-p.get("signal", 0), p["date"]))
    return fresh[0]


def mark_prototyped(problem_id: str, repo_url: str) -> None:
    problems = load_problems()
    for p in problems:
        if p["id"] == problem_id:
            p["status"] = "prototyped"
            p["prototype_repo"] = repo_url
    save_problems(problems)


# ── Generation ───────────────────────────────────────────────────────────────

# Kept deliberately short to minimise input tokens (every token counts).
SYSTEM_PROMPT = (
    "You are AutoScout, an automated prototype generator specializing in "
    "agentic AI systems: agent orchestration, memory, evaluation/"
    "observability, guardrails/security, multi-agent coordination, tool-"
    "calling/MCP protocols, and agent ops/deployment. Output a working, self-"
    "contained project in whatever language and project shape genuinely fits "
    "the problem best — do not default to a Python script out of habit. "
    "Be concise — prioritise clarity, correctness, and real usability as a "
    "starting point for a project someone would actually build on."
)

FORMAT_RULES = """\
Decide the project shape yourself — Python package, Node/TypeScript CLI, Go \
service, MCP server, browser or VS Code extension, spec plus reference \
implementation, or anything else that is the best fit. Do not force Python \
or any fixed file layout if another shape suits the problem better.

Decide independently whether the prototype's core feature genuinely requires \
calling an LLM:
- If yes: call the Gemini API (google-genai SDK for Python, or the plain \
REST endpoint for other languages/stacks) and read the key from the \
GEMINI_API_KEY environment variable. Do not hardcode a key.
- If no (e.g. the idea is orchestration, memory storage, tracing, config, a \
protocol/format — pure tooling with no inherent LLM call): do not wire in \
any LLM API at all. It should run standalone with zero API keys.

Output format — one header line per file, in EXACTLY this form, with the \
real filename/path substituted in (never the literal word "path"):

=== <filename-or-relative-path> ===
<the file's full content>

For example, a Node/TS CLI would start:

=== package.json ===
{{ ... }}

=== src/index.ts ===
...

Choose filenames/paths that fit the chosen stack (e.g. requirements.txt for \
Python, package.json for Node/TS, go.mod for Go, Dockerfile for a service). \
No markdown fences inside any file's content. Always include a README.md as \
the FIRST section covering:
- The real problem and who it affects
- Why this project shape/stack was chosen for it
- Setup and usage instructions
- A line stating plainly whether GEMINI_API_KEY is required to run it

Keep it tight: 3-7 files total, working code, no filler.
"""

USER_TEMPLATE = (
    "Generate an agentic-AI prototype project for: {topic}\n\n" + FORMAT_RULES
)

PROBLEM_TEMPLATE = (
    "Generate a prototype that addresses this real problem observed in the "
    "agentic-AI community:\n\n"
    "Problem: {title}\n"
    "Details: {problem}\n\n" + FORMAT_RULES
)


def generate_prototype_files(api_key: str, prompt: str) -> dict[str, str]:
    try:
        raw = call_gemini(api_key, prompt, SYSTEM_PROMPT, max_tokens=MAX_OUTPUT_TOKENS)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    files = parse_sections(raw)
    if not files:
        print("ERROR: no sections found in model response.", file=sys.stderr)
        print(raw[:1000], file=sys.stderr)
        sys.exit(1)
    return files


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

    today = date.today()
    problem = pick_problem()
    if problem:
        topic = problem["title"]
        prompt = PROBLEM_TEMPLATE.format(title=problem["title"],
                                         problem=problem["problem"])
    else:
        topic = pick_topic()
        prompt = USER_TEMPLATE.format(topic=topic)

    repo_name = f"{slugify(topic)}-{today.isoformat()}"
    owner = get_authenticated_user(gh_token)
    repo_url = f"https://github.com/{owner}/{repo_name}"
    scouted_today = sum(1 for p in load_problems() if p["date"] == today.isoformat())

    if repo_exists(owner, repo_name, gh_token):
        print(f"Repo {owner}/{repo_name} already exists — nothing to do.")
        if problem:
            mark_prototyped(problem["id"], repo_url)
        update_section(gh_token, "generate",
                      f"**New repo**: [{repo_name}]({repo_url}) (already existed this run)\n"
                      f"**Source**: {'scouted problem' if problem else 'fallback topic pool'}\n"
                      f"**Topic**: {topic}\n\n"
                      f"Problems scouted today: {scouted_today}")
        sys.exit(0)

    print(f"─── AutoScout: generating 1 project ───")
    print(f"Source     : {'scouted problem' if problem else 'fallback topic pool'}")
    print(f"Topic      : {topic}")
    print(f"New repo   : {owner}/{repo_name}")
    print(f"Tokens     : up to {MAX_OUTPUT_TOKENS} output  "
          f"(~{MAX_OUTPUT_TOKENS/250000*100:.1f}% of 250k daily budget)")

    files = generate_prototype_files(gemini_key, prompt)

    repo = create_repo(repo_name, gh_token)
    push_prototype(repo["clone_url"], files, gh_token)
    add_repo(f"{owner}/{repo_name}", repo_name, today.isoformat(), topic)

    if problem:
        mark_prototyped(problem["id"], repo_url)
        print(f"Marked problem '{problem['id']}' as prototyped.")

    update_section(gh_token, "generate",
                  f"**New repo**: [{repo_name}]({repo_url})\n"
                  f"**Source**: {'scouted problem' if problem else 'fallback topic pool'}\n"
                  f"**Topic**: {topic}\n\n"
                  f"Problems scouted today: {scouted_today}")

    print(f"\nDone — {repo_url}")


if __name__ == "__main__":
    main()
