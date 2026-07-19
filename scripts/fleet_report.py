#!/usr/bin/env python3
"""Generate FLEET.md — a daily health dashboard of every AutoScout repo.

Merges this repo's registry (Gemini maturation state) with
AutoScout-Engine's registry (Groq advancement state, fetched from its
public raw URL) into one table: stars, pass counts, last activity, and
which rotation tier each repo currently sits in.
"""

import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from mature_repo import is_active
from repo_registry import load_registry

REPO_ROOT = Path(__file__).parent.parent.resolve()
ENGINE_REGISTRY_URL = ("https://raw.githubusercontent.com/sathiya-22/"
                       "AutoScout-Engine/main/state/registry.jsonl")


def fetch_engine_registry() -> dict[str, dict]:
    req = urllib.request.Request(ENGINE_REGISTRY_URL)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"WARN: could not fetch Engine registry: {str(e)[:120]}")
        return {}
    entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {e["full_name"]: e for e in entries}


def render(lab: list[dict], engine: dict[str, dict]) -> str:
    today = date.today()
    rows = []
    for e in sorted(lab, key=lambda e: (-e.get("stars", 0), e.get("created", "")),
                    ):
        eng = engine.get(e["full_name"], {})
        last_touched = max(e.get("last_matured") or "", eng.get("last_reviewed") or "") or "—"
        tier = "🟢 active" if is_active(e, today) else "💤 dormant"
        name = e["name"]
        rows.append(
            f"| [{name}](https://github.com/{e['full_name']}) "
            f"| {e.get('created', '—')} | {e.get('stars', 0)} "
            f"| {e.get('iterations', 0)} | {eng.get('advancement_passes', 0)} "
            f"| {last_touched} | {tier} |"
        )
    total_gemini = sum(e.get("iterations", 0) for e in lab)
    total_groq = sum(e.get("advancement_passes", 0) for e in engine.values())
    return (
        f"# AutoScout Fleet\n\n"
        f"Auto-generated daily — {len(lab)} repos, "
        f"{total_gemini} Gemini maturation passes, "
        f"{total_groq} Groq advancement passes. "
        f"Last updated {today.isoformat()}.\n\n"
        f"🟢 active = starred or <14 days old, rotates freely · "
        f"💤 dormant = revisited monthly\n\n"
        f"| Repo | Created | ⭐ | Gemini | Groq | Last touched | Tier |\n"
        f"|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
    )


def main() -> None:
    lab = load_registry()
    if not lab:
        print("Registry empty — nothing to report.")
        return
    engine = fetch_engine_registry()
    content = render(lab, engine)
    out = REPO_ROOT / "FLEET.md"
    if out.exists() and out.read_text(encoding="utf-8") == content:
        print("FLEET.md unchanged.")
        return
    out.write_text(content, encoding="utf-8")
    print(f"FLEET.md updated ({len(lab)} repos).")


if __name__ == "__main__":
    main()
