# AutoScout-Lab

Orchestrator for AutoScout — an automated pipeline that scouts real problems
in the agentic-AI ecosystem (agent orchestration, memory, evals/
observability, guardrails/security, multi-agent coordination, tool-calling/
MCP protocols, agent ops) and turns them into prototypes that keep maturing
over time.

## Pipeline

1. **Daily problem scout** (`scripts/scout_problems.py`, 09:00 IST) — pulls
   agentic-AI pain signals from Hacker News, GitHub issues (keyword search
   plus targeted searches of core agent-framework repos), and Stack Overflow,
   distills them with Gemini into structured problem statements, and appends
   them to [`problems/problem_log.jsonl`](problems/problem_log.jsonl).
2. **Daily prototype generator** (`scripts/generate_prototype.py`) — picks the
   highest-signal unaddressed problem from the log and generates a prototype
   targeting it, in whatever project shape/stack genuinely fits (Python,
   Node/TS, Go, MCP server, ...) — published as its own standalone public repo
   (`<problem-slug>-<date>`) and registered in
   [`repos/registry.jsonl`](repos/registry.jsonl). Falls back to a static
   topic pool if the log is empty. Only wires in `GEMINI_API_KEY` when the
   idea's core feature genuinely needs an LLM call.
3. **Repo maturation loop** (`scripts/mature_repo.py`, 09:30 IST) — every day,
   advances exactly ONE previously-generated repo by one meaningful increment
   (a real feature, tests, docs, packaging, a bug fix), committed straight to
   that repo's `main` and logged in its own `MATURATION_LOG.md`. Repos rotate
   never-matured-first then oldest-last-matured-first, so this adds a flat
   +1 Gemini call/day forever regardless of how many repos exist — no repo is
   left behind, none blow the shared free-tier quota.
4. **Weekly major-problem synthesis** (`scripts/weekly_synthesis.py`, Sundays)
   — ranks the week's problems, picks the most significant one, writes a build
   spec to `major/`, and opens an issue on this repo as the signal to build a
   serious open-source solution for it.

The `ai_scout_batch_*` folders are historical batches from before the
one-repo-per-day setup.
