#!/usr/bin/env python3
"""Minimal Groq call wrapper, dedicated to the research/eval stages.

AutoScout-Lab's main maturation loop runs on Gemini. The research stage
(research.py) and eval-judge scoring (eval_harness.py) intentionally use a
SEPARATE Groq key (GROQ_RESEARCH_API_KEY) so their extra calls never compete
with the main maturation budget or quota — a quota-isolation choice, not a
capability one. Kept as its own tiny module rather than pulling in
AutoScout-Engine's groq_common.py, since Lab and Engine are deliberately
independent (see registry.py's docstring in Engine)."""

import json
import sys
import time
import urllib.error
import urllib.request

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 2
RETRY_BASE_SEC = 20


def _is_transient(err: str) -> bool:
    if any(x in err for x in ("404", "403", "401")):
        return False
    keywords = ("429", "503", "rate limit", "overloaded", "unavailable", "retry")
    return any(k.lower() in err.lower() for k in keywords)


def call_groq(api_key: str, prompt: str, system: str, max_tokens: int) -> str:
    """One prompt -> response text via Groq's OpenAI-compatible endpoint."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }).encode()

    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(GROQ_API_URL, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        # Groq sits behind Cloudflare, which blocks urllib's default
        # "Python-urllib/x.x" User-Agent as a bot signature (403, code 1010).
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; AutoScout-Lab/1.0)")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"]
                if not text:
                    raise RuntimeError("empty response")
                return text
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:300]
            err = f"{e.code} {err_body}"
        except Exception as e:
            err = str(e)

        if _is_transient(err) and attempt < MAX_RETRIES:
            wait = RETRY_BASE_SEC * (2 ** (attempt - 1))
            print(f"  Transient error (attempt {attempt}/{MAX_RETRIES}), "
                 f"retrying in {wait}s... [{err[:150]}]", flush=True)
            time.sleep(wait)
        else:
            print(f"  Groq call failed after {attempt} attempt(s): {err[:200]}",
                 file=sys.stderr)
            break

    raise RuntimeError("Groq API call exhausted retries")
