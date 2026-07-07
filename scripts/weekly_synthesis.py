#!/usr/bin/env python3
"""AutoScout weekly major-problem synthesis (runs Sundays).

Reads the past week's scouted problems, asks Gemini to pick the single most
significant one and draft a build spec, writes major/<year>-W<week>-major-problem.md,
marks the problem's status "major", and opens a GitHub issue on this repo as
the signal to build the real solution that week.

Budget: 1 Gemini request per run (~4k output tokens).
"""

import json
import os
import sys
import urllib.request
from datetime import date, timedelta

from scout_common import (REPO_ROOT, call_gemini, load_problems,
                          parse_json_lenient, save_problems)

MAX_CANDIDATES = 40

SYSTEM_PROMPT = (
    "You are AutoScout's chief analyst. You select the most impactful real "
    "problem of the week and draft a practical build spec for a small "
    "open-source project."
)

USER_TEMPLATE = """\
Below are AI-domain problems scouted from the community this week (JSON).

Pick the ONE most significant problem worth building a serious open-source \
solution for. Judge by: how many people it affects, severity, absence of good \
existing solutions, and feasibility for a small project.

Return JSON:
{{"chosen_id": "<id of the chosen problem>",
  "rationale": "why this one (3-4 sentences)",
  "spec_markdown": "build spec in markdown with sections: ## Problem, \
## Evidence, ## Proposed solution, ## MVP scope, ## Milestones"}}

Problems:
{problems}
"""


def create_issue(title: str, body: str) -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("SCOUT_PAT")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not (token and repo):
        print("WARN: GITHUB_TOKEN/GITHUB_REPOSITORY missing — skipping issue.",
              file=sys.stderr)
        return None
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({"title": title, "body": body}).encode(),
        method="POST",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("html_url")
    except Exception as e:
        print(f"WARN: issue creation failed: {str(e)[:200]}", file=sys.stderr)
        return None


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    problems = load_problems()
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    candidates = [p for p in problems
                  if p["date"] >= cutoff and p.get("status") != "major"]
    candidates.sort(key=lambda p: -p.get("signal", 0))
    candidates = candidates[:MAX_CANDIDATES]

    if not candidates:
        print("No problems scouted this week — nothing to synthesize.")
        return

    print(f"─── AutoScout: weekly synthesis over {len(candidates)} problem(s) ───")
    raw = call_gemini(
        api_key,
        USER_TEMPLATE.format(problems=json.dumps(candidates, ensure_ascii=False)),
        SYSTEM_PROMPT,
        max_tokens=4000,
        json_mode=True,
    )
    result = parse_json_lenient(raw)

    chosen = next((p for p in candidates if p["id"] == result.get("chosen_id")),
                  candidates[0])  # fall back to highest signal
    rationale = (result.get("rationale") or "").strip()
    spec = (result.get("spec_markdown") or "").strip()

    iso_year, iso_week, _ = date.today().isocalendar()
    report_rel = f"major/{iso_year}-W{iso_week:02d}-major-problem.md"
    report_path = REPO_ROOT / report_rel
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sources = "\n".join(f"- {u}" for u in chosen.get("sources", [])) or "- (none logged)"
    proto = chosen.get("prototype_repo")
    proto_line = f"\nDaily prototype: {proto}\n" if proto else ""
    report_path.write_text(
        f"# Major problem of week {iso_year}-W{iso_week:02d}\n\n"
        f"**{chosen['title']}**  (id: `{chosen['id']}`, signal: {chosen.get('signal', 0)})\n\n"
        f"{chosen.get('problem', '')}\n\n"
        f"## Why this one\n\n{rationale}\n\n"
        f"## Sources\n\n{sources}\n{proto_line}\n"
        f"---\n\n{spec}\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_rel}")

    for p in problems:
        if p["id"] == chosen["id"]:
            p["status"] = "major"
            p["major_week"] = f"{iso_year}-W{iso_week:02d}"
    save_problems(problems)

    issue_url = create_issue(
        f"Build week {iso_year}-W{iso_week:02d}: {chosen['title']}",
        f"AutoScout picked this week's major problem.\n\n"
        f"**Problem:** {chosen.get('problem', '')}\n\n"
        f"**Rationale:** {rationale}\n\n"
        f"Full spec: [`{report_rel}`](../blob/main/{report_rel})\n",
    )
    if issue_url:
        print(f"Opened issue: {issue_url}")

    print(f"\nDone — major problem: {chosen['title']}")


if __name__ == "__main__":
    main()
