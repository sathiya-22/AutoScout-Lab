#!/usr/bin/env python3
"""Registry of every AutoScout-generated prototype repo.

Used by generate_prototype.py (to register each new repo at creation time)
and mature_repo.py (to pick the repo most overdue for a maturation cycle).
sync_registry() self-heals against GitHub's actual repo list — any
AutoScout repo missing from the registry gets added, and any registry entry
whose repo no longer exists (e.g. manually deleted) gets dropped.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
REGISTRY_PATH = REPO_ROOT / "repos" / "registry.jsonl"
GITHUB_API = "https://api.github.com"
# Repo descriptions are now per-repo/topic-specific (for public "building in
# public" polish), so discovery matches on a stable prefix instead of an
# exact string. The old fixed string is also matched for repos created
# before this change.
AUTOSCOUT_DESCRIPTION_PREFIX = "AutoScout AI-generated prototype:"
AUTOSCOUT_DESCRIPTION_LEGACY = "Auto-generated AI prototype by AutoScout"


def _is_autoscout_repo(description: str | None) -> bool:
    description = description or ""
    return (description.startswith(AUTOSCOUT_DESCRIPTION_PREFIX)
            or description == AUTOSCOUT_DESCRIPTION_LEGACY)


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return [json.loads(line) for line in
            REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_registry(entries: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )


def add_repo(full_name: str, name: str, created: str, topic: str) -> None:
    entries = load_registry()
    if any(e["full_name"] == full_name for e in entries):
        return
    entries.append({
        "full_name": full_name,
        "name": name,
        "created": created,
        "topic": topic,
        "iterations": 0,
        "last_matured": None,
    })
    save_registry(entries)


def _gh_get(path: str, token: str):
    req = urllib.request.Request(f"{GITHUB_API}{path}")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"GitHub API GET {path} failed: {e.code}") from e


def sync_registry(owner: str, token: str) -> list[dict]:
    """Reconcile the registry against GitHub's actual repo list."""
    entries = {e["full_name"]: e for e in load_registry()}
    live_full_names: set[str] = set()

    page = 1
    while True:
        repos = _gh_get(
            f"/users/{owner}/repos?per_page=100&page={page}&sort=created&direction=asc",
            token,
        )
        if not repos:
            break
        for r in repos:
            if not _is_autoscout_repo(r.get("description")):
                continue
            full_name = r["full_name"]
            live_full_names.add(full_name)
            if full_name not in entries:
                entries[full_name] = {
                    "full_name": full_name,
                    "name": r["name"],
                    "created": r["created_at"][:10],
                    "topic": r["name"].rsplit("-", 3)[0].replace("-", " "),
                    "iterations": 0,
                    "last_matured": None,
                }
            # Refreshed on every sync — outside interest is the liveliness
            # signal the tiered rotation prioritizes by.
            entries[full_name]["stars"] = r.get("stargazers_count", 0)
        if len(repos) < 100:
            break
        page += 1

    pruned = [e for full_name, e in entries.items() if full_name in live_full_names]
    save_registry(pruned)
    return pruned
