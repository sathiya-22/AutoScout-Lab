#!/usr/bin/env python3
"""AutoScout per-repo maturation loop.

Every run advances exactly ONE previously-generated prototype repo by one
increment — the single most valuable next step toward a genuinely complete,
production-quality project (a real feature, error handling, tests, docs,
packaging, or a genuine bug fix) — and commits it straight to that repo's
main branch. Repos are visited in rotation: never-matured repos first
(oldest created first), then oldest-last-matured first. This keeps the
Gemini budget flat at +1 request/day forever, no matter how many repos
AutoScout has generated.

Each matured repo carries its own MATURATION_LOG.md so the model has
continuity across cycles without needing this orchestrator to remember
anything beyond the iteration count.

Budget: 1 Gemini request per run (~7k output tokens).
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from repo_registry import load_registry, save_registry, sync_registry
from scout_common import call_gemini, parse_sections

GITHUB_API = "https://api.github.com"
MAX_OUTPUT_TOKENS = 7000

MAX_FILES_READ = 40
MAX_FILE_BYTES = 20_000        # skip absurdly large files
MAX_CONTEXT_CHARS = 60_000     # cap total repo context fed to Gemini

GROWTH_LOG = "MATURATION_LOG.md"
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
SKIP_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".lock")


# ── GitHub helpers ───────────────────────────────────────────────────────────

def _gh(method: str, path: str, token: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{GITHUB_API}{path}", data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"GitHub API {method} {path} failed: {e.code} "
                           f"{e.read().decode()[:300]}") from e


def get_authenticated_user(token: str) -> str:
    return _gh("GET", "/user", token)["login"]


def fetch_repo_context(full_name: str, token: str) -> dict[str, str] | None:
    """Fetch text file contents via the Git Trees API — no clone needed."""
    tree = _gh("GET", f"/repos/{full_name}/git/trees/main?recursive=1", token)
    if not tree:
        return None
    files: dict[str, str] = {}
    total = 0
    for item in tree.get("tree", []):
        if item["type"] != "blob":
            continue
        path = item["path"]
        if any(part in SKIP_DIRS for part in path.split("/")):
            continue
        if path.endswith(SKIP_EXTS):
            continue
        if item.get("size", 0) > MAX_FILE_BYTES:
            continue
        if len(files) >= MAX_FILES_READ or total >= MAX_CONTEXT_CHARS:
            break
        blob = _gh("GET", f"/repos/{full_name}/git/blobs/{item['sha']}", token)
        if not blob:
            continue
        try:
            content = base64.b64decode(blob["content"]).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
        files[path] = content
        total += len(content)
    return files


# ── Rotation ─────────────────────────────────────────────────────────────────

def pick_due_repo(registry: list[dict]) -> dict | None:
    """Never-matured repos first (oldest created first), then oldest
    last_matured first — this is what keeps the daily budget flat."""
    if not registry:
        return None
    return sorted(registry,
                  key=lambda r: (r.get("last_matured") or "0000-00-00",
                                 r.get("created", "9999-99-99")))[0]


# ── Generation ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are AutoScout's maturation engineer. You take an early-stage "
    "agentic-AI prototype and advance it by exactly ONE meaningful increment "
    "per cycle — a real feature, error handling, input validation, tests, "
    "docs, packaging, or a genuine bug fix — without breaking what already "
    "works. Never regenerate the whole project from scratch; only output "
    "files you are creating or changing.\n\n"
    "Hard rules:\n"
    "1. NEVER replace real, working logic (a real API call, a real "
    "computation) with a stub, mock, placeholder, or simulated/fake data — "
    "that is a regression, not an increment, even if the surrounding code "
    "looks cleaner.\n"
    "2. The growth log is exactly what is shown to you. If it says 'none "
    "yet', do not invent any prior entries or echo that placeholder text "
    "into the file — write ONLY the one new line, using the exact date "
    "given, not a guessed or remembered one.\n"
    "3. If you introduce a new import/package, you MUST also update the "
    "dependency manifest (requirements.txt, package.json, go.mod, etc.) in "
    "the same response — an import with no declared dependency is broken."
)

MATURE_TEMPLATE = """\
This repository already exists and needs ONE incremental improvement toward \
becoming a genuinely complete, production-quality project.

Repo: {full_name}
Original topic/problem: {topic}
Iteration: {iteration}
Today's date (use this exact date in the log, do not guess another one): {today}

Growth log so far — this is the COMPLETE history, verbatim, nothing is \
hidden from you:
{growth_log}

Current files:
{file_dump}

Pick the SINGLE most valuable next increment. Output ONLY the files you are \
creating or modifying — leave everything else untouched. Do not remove or \
fake any existing real functionality (see hard rule 1). Use this exact \
format, one header per file, with the real filename/path substituted in:

=== <filename-or-relative-path> ===
<the file's full new content>

You MUST include an updated === {growth_log_name} === with exactly ONE new \
bullet appended, dated {today} — keep every prior line exactly as shown \
above, and do not add any entry that isn't shown above plus this one new one.

No markdown fences inside any file's content.
"""


def build_prompt(entry: dict, files: dict[str, str]) -> str:
    growth_log = files.get(GROWTH_LOG,
                           "(none yet — this is truly the first maturation cycle; "
                           "do not invent any earlier entries.)")
    dump = "\n\n".join(f"----- FILE: {path} -----\n{content}"
                       for path, content in files.items())
    return MATURE_TEMPLATE.format(
        full_name=entry["full_name"],
        topic=entry.get("topic", entry["name"]),
        iteration=entry.get("iterations", 0) + 1,
        today=date.today().isoformat(),
        growth_log=growth_log,
        file_dump=dump,
        growth_log_name=GROWTH_LOG,
    )


def sanitize_log(old_log: str, model_log: str, today: str) -> str:
    """Rebuild the log deterministically instead of trusting the model to
    reproduce prior lines unchanged and not invent/echo placeholder text:
    keep the real old_log verbatim, and append ONLY genuinely-new lines
    dated today. Gemini fabricated a wrong-dated entry AND, separately,
    leaked the prompt's '(none yet...)' placeholder text verbatim into a
    committed file in testing — prompting alone isn't reliable enough."""
    old_lines = set(old_log.splitlines())
    new_dated_lines = [line for line in model_log.splitlines()
                      if line.strip() and line not in old_lines and today in line]
    if not new_dated_lines:
        return old_log  # model didn't produce a valid dated line — caller falls back
    return (old_log.rstrip("\n") + "\n" if old_log.strip() else "") + "\n".join(new_dated_lines)


def commit_summary(old_log: str, new_log: str) -> str:
    old_lines = set(old_log.splitlines())
    for line in new_log.splitlines():
        if line.strip() and line not in old_lines:
            return line.strip("- ").strip()[:72]
    return "maturation increment"


# ── Apply changes ────────────────────────────────────────────────────────────

def push_maturation(full_name: str, files: dict[str, str], token: str,
                    iteration: int, summary: str) -> None:
    authed_url = f"https://x-access-token:{token}@github.com/{full_name}.git"
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp)

        def run(*args: str) -> None:
            subprocess.run(args, cwd=repo_dir, check=True)

        subprocess.run(["git", "clone", "--depth", "1", "-q", authed_url, str(repo_dir)],
                       check=True)
        run("git", "config", "user.name", "AutoScout Bot")
        run("git", "config", "user.email", "autoscout-bot@users.noreply.github.com")

        for filename, content in files.items():
            target = repo_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
            print(f"  wrote {filename}  ({len(content):,} chars)")

        run("git", "add", "-A")
        run("git", "commit", "-q", "-m",
           f"chore(autoscout): maturation iteration {iteration} — {summary}")
        run("git", "push", "-q")


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

    print("─── AutoScout: per-repo maturation ───")
    owner = get_authenticated_user(gh_token)
    registry = sync_registry(owner, gh_token)
    print(f"Registry   : {len(registry)} tracked repo(s)")

    if not registry:
        print("No repos to mature yet.")
        return

    entry = pick_due_repo(registry)
    full_name = entry["full_name"]
    print(f"Due repo   : {full_name}  (iteration {entry.get('iterations', 0) + 1}, "
         f"last matured: {entry.get('last_matured') or 'never'})")

    files = fetch_repo_context(full_name, gh_token)
    if files is None:
        print(f"WARN: {full_name} has no 'main' branch content — "
             "dropping from registry.", file=sys.stderr)
        registry = [e for e in registry if e["full_name"] != full_name]
        save_registry(registry)
        return

    old_growth_log = files.get(GROWTH_LOG, "")
    prompt = build_prompt(entry, files)

    try:
        raw = call_gemini(gemini_key, prompt, SYSTEM_PROMPT, max_tokens=MAX_OUTPUT_TOKENS)
    except RuntimeError as e:
        print(f"ERROR: {e} — leaving registry untouched for a retry next run.",
             file=sys.stderr)
        sys.exit(1)

    edited = parse_sections(raw)
    if not edited:
        print("ERROR: no sections found in model response — retry next run.",
             file=sys.stderr)
        print(raw[:1000], file=sys.stderr)
        sys.exit(1)

    iteration = entry.get("iterations", 0) + 1
    today = date.today().isoformat()
    model_growth_log = edited.get(GROWTH_LOG, "")
    new_growth_log = sanitize_log(old_growth_log, model_growth_log, today) \
        if model_growth_log else old_growth_log
    if new_growth_log == old_growth_log:
        # model gave no usable dated line — fall back to a generic one so
        # the log still records that an iteration happened
        new_growth_log = (old_growth_log.rstrip("\n") + "\n" if old_growth_log.strip() else "") + \
                         f"- {today}: iteration {iteration} (see commit for details)"
    edited[GROWTH_LOG] = new_growth_log

    summary = commit_summary(old_growth_log, new_growth_log)

    try:
        push_maturation(full_name, edited, gh_token, iteration, summary)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: failed to push to {full_name}: {e} — "
             "leaving registry untouched for a retry next run.", file=sys.stderr)
        sys.exit(1)

    entry["iterations"] = iteration
    entry["last_matured"] = date.today().isoformat()
    save_registry(registry)

    print(f"\nDone — {full_name} advanced to iteration {iteration}: {summary}")


if __name__ == "__main__":
    main()
