#!/usr/bin/env python3
"""AutoScout daily problem scout.

Gathers real pain signals in the agentic-AI space — agent orchestration,
memory, evaluation/observability, guardrails/security, multi-agent
coordination, tool-calling/MCP protocols, agent ops/deployment — from Hacker
News, GitHub issues (both keyword search and targeted agent-framework repos)
and Stack Overflow. Distills them into structured problem statements with one
Gemini call and appends new ones to problems/problem_log.jsonl.

Budget: 1 Gemini request per run (~4k output tokens).
Run with --dry-run to fetch and print signals without calling Gemini or
writing the log.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

from scout_common import (call_gemini, load_problems, parse_json_lenient,
                          save_problems, slugify)

LOOKBACK_DAYS = 2       # overlap between runs; dedup handles repeats
MAX_SIGNALS = 60        # cap prompt size
MAX_NEW_PROBLEMS = 5
USER_AGENT = "AutoScout-Lab problem scout (github.com/sathiya-22/AutoScout-Lab)"

# Agentic-AI-ops specific search terms — orchestration, memory, evals,
# observability, guardrails, multi-agent, tool-calling/MCP, deployment.
AGENT_KEYWORDS = (
    "agent orchestration", "agent memory", "MCP server", "multi-agent",
    "agent observability", "agent guardrails", "tool calling",
    "agent evaluation",
)

# High-signal open-source repos at the center of the agent ecosystem —
# searched directly for their most-reacted-to open issues.
AGENT_FRAMEWORK_REPOS = (
    "langchain-ai/langchain",
    "langchain-ai/langgraph",
    "crewAIInc/crewAI",
    "microsoft/autogen",
    "openai/openai-agents-python",
    "modelcontextprotocol/servers",
    "modelcontextprotocol/python-sdk",
)


# ── Signal sources (each failure-tolerant: one dead API must not kill the run) ─

def _get_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  WARN: fetch failed for {url.split('?')[0]}: {str(e)[:120]}",
              file=sys.stderr)
        return None


def fetch_hackernews(since_ts: int) -> list[dict]:
    signals = []
    for query in AGENT_KEYWORDS:
        params = urllib.parse.urlencode({
            "query": query,
            "tags": "story",
            "numericFilters": f"points>10,created_at_i>{since_ts}",
            "hitsPerPage": 15,
        })
        # search_by_date: the relevance-sorted index rejects points filters
        data = _get_json(f"https://hn.algolia.com/api/v1/search_by_date?{params}")
        for hit in (data or {}).get("hits", []):
            signals.append({
                "source": "hackernews",
                "title": hit.get("title") or "",
                "url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
                "score": (hit.get("points") or 0) + (hit.get("num_comments") or 0),
            })
    return signals


def _search_github_issues(q: str, headers: dict, source: str) -> list[dict]:
    params = urllib.parse.urlencode(
        {"q": q, "sort": "reactions", "order": "desc", "per_page": 15})
    data = _get_json(f"https://api.github.com/search/issues?{params}", headers)
    signals = []
    for item in (data or {}).get("items", []):
        reactions = (item.get("reactions") or {}).get("total_count", 0)
        signals.append({
            "source": source,
            "title": item.get("title") or "",
            "url": item.get("html_url") or "",
            "score": reactions + item.get("comments", 0),
        })
    return signals


def fetch_github_issues(since: str, token: str) -> list[dict]:
    """Keyword search across all of GitHub for recent agent-related issues."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    signals = []
    for keyword in AGENT_KEYWORDS:
        q = f'"{keyword}" is:issue is:open created:>{since} reactions:>2'
        signals += _search_github_issues(q, headers, "github")
        time.sleep(2)  # search API rate limit is per-minute; stay polite
    return signals


def fetch_github_agent_repos(token: str) -> list[dict]:
    """Most-reacted-to open issues directly from core agent-framework repos —
    sharper, pre-validated signal than a general keyword search."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    signals = []
    for repo in AGENT_FRAMEWORK_REPOS:
        q = f"is:issue is:open reactions:>1 repo:{repo}"
        signals += _search_github_issues(q, headers, "github:" + repo)
        time.sleep(2)
    return signals


def fetch_stackoverflow(since_ts: int) -> list[dict]:
    signals = []
    for tag in ("langchain", "langgraph", "openai-api", "autogen"):
        params = urllib.parse.urlencode({
            "order": "desc", "sort": "votes", "tagged": tag,
            "site": "stackoverflow", "fromdate": since_ts, "pagesize": 10,
        })
        data = _get_json(f"https://api.stackexchange.com/2.3/questions?{params}")
        for item in (data or {}).get("items", []):
            signals.append({
                "source": "stackoverflow",
                "title": item.get("title") or "",
                "url": item.get("link") or "",
                "score": item.get("score", 0) + item.get("answer_count", 0),
            })
    return signals


# ── Distillation ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are AutoScout's problem analyst. You identify real, recurring problems "
    "developers face building and operating agentic AI systems — agent "
    "orchestration, agent memory, evaluation/observability, guardrails and "
    "security, multi-agent coordination, tool-calling/MCP protocols, and agent "
    "ops/deployment — from community signals."
)

USER_TEMPLATE = """\
From the community signals below (source | engagement | title | url), extract \
up to {n} DISTINCT real problems developers face building or operating \
agentic AI systems (agent orchestration, memory, evals, observability, \
guardrails/security, multi-agent coordination, tool-calling/MCP, agent ops).

Only include genuine pain points (bugs, missing tooling, recurring frustrations, \
unmet needs) — skip product announcements, news and opinion pieces, and skip \
anything that isn't really about agentic systems (e.g. generic model-quality \
complaints with no agent/tooling angle).

Return a JSON array; each element:
{{"title": "short problem name (max 8 words)",
  "problem": "2-3 sentences: what the problem is and who it affects",
  "source_urls": ["the signal urls that evidence this problem"],
  "signal": <integer: sum of engagement scores of the sources used>}}

Signals:
{signals}
"""


def distill(api_key: str, signals: list[dict]) -> list[dict]:
    lines = "\n".join(
        f"{s['source']} | {s['score']} | {s['title']} | {s['url']}"
        for s in signals
    )
    raw = call_gemini(
        api_key,
        USER_TEMPLATE.format(n=MAX_NEW_PROBLEMS, signals=lines),
        SYSTEM_PROMPT,
        max_tokens=4000,
        json_mode=True,
    )
    parsed = parse_json_lenient(raw)
    return parsed if isinstance(parsed, list) else []


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    dry_run = "--dry-run" in sys.argv

    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=LOOKBACK_DAYS)
    since_ts = int(since_dt.timestamp())
    since_day = since_dt.date().isoformat()

    print("─── AutoScout: scouting real agentic-AI problems ───")
    scout_pat = os.environ.get("SCOUT_PAT", "")
    signals = (
        fetch_hackernews(since_ts)
        + fetch_github_issues(since_day, scout_pat)
        + fetch_github_agent_repos(scout_pat)
        + fetch_stackoverflow(since_ts)
    )
    signals = [s for s in signals if s["title"] and s["url"]]
    signals.sort(key=lambda s: -s["score"])
    signals = signals[:MAX_SIGNALS]

    by_source: dict[str, int] = {}
    for s in signals:
        by_source[s["source"]] = by_source.get(s["source"], 0) + 1
    print(f"Signals    : {len(signals)}  {by_source}")

    if dry_run:
        for s in signals[:20]:
            print(f"  [{s['source']:>13}] {s['score']:>4}  {s['title'][:80]}")
        return

    if not signals:
        print("WARN: no signals fetched — skipping distillation.", file=sys.stderr)
        return

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    candidates = distill(api_key, signals)

    problems = load_problems()
    known_ids = {p["id"] for p in problems}
    added = 0
    for c in candidates:
        title = (c.get("title") or "").strip()
        if not title:
            continue
        pid = slugify(title)
        if pid in known_ids:
            continue
        problems.append({
            "id": pid,
            "date": date.today().isoformat(),
            "title": title,
            "problem": (c.get("problem") or "").strip(),
            "sources": c.get("source_urls") or [],
            "signal": int(c.get("signal") or 0),
            "status": "new",
        })
        known_ids.add(pid)
        added += 1
        print(f"  + {title}  (signal {c.get('signal', 0)})")

    save_problems(problems)
    print(f"\nDone — {added} new problem(s), {len(problems)} total in log.")


if __name__ == "__main__":
    main()
