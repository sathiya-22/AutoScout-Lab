# AutoScout-Lab

Orchestrator for AutoScout — an automated pipeline that scouts real problems
in the AI/LLM ecosystem and turns them into prototypes and projects.

## Pipeline

1. **Daily problem scout** (`scripts/scout_problems.py`, 09:00 IST) — pulls
   pain signals from Hacker News, GitHub issues and Stack Overflow, distills
   them with Gemini into structured problem statements, and appends them to
   [`problems/problem_log.jsonl`](problems/problem_log.jsonl).
2. **Daily prototype generator** (`scripts/generate_prototype.py`) — picks the
   highest-signal unaddressed problem from the log and generates a prototype
   targeting it, published as its own standalone public repo
   (`<problem-slug>-<date>`). Falls back to a static topic pool if the log is
   empty.
3. **Weekly major-problem synthesis** (`scripts/weekly_synthesis.py`, Sundays)
   — ranks the week's problems, picks the most significant one, writes a build
   spec to `major/`, and opens an issue on this repo as the signal to build a
   serious open-source solution for it.

The `ai_scout_batch_*` folders are historical batches from before the
one-repo-per-day setup.
