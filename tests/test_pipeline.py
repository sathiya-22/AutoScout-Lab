#!/usr/bin/env python3
"""Unit tests for AutoScout-Lab's pipeline logic.

Pure-logic only — no network, no Gemini, no filesystem side effects.
Run: python3 -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_prototype import derive_topics  # noqa: E402
from mature_repo import commit_summary, pick_due_repo, sanitize_log  # noqa: E402
from scout_common import (broken_python_files, parse_json_lenient,  # noqa: E402
                          parse_sections, slugify)
from scout_problems import find_near_duplicate  # noqa: E402


class TestParseSections(unittest.TestCase):
    def test_multi_file(self):
        raw = "=== README.md ===\nhi\n\n=== src/x.py ===\nprint(1)\n"
        self.assertEqual(set(parse_sections(raw)), {"README.md", "src/x.py"})

    def test_unsafe_paths_dropped(self):
        raw = ("=== ../escape.txt ===\nbad\n\n=== /tmp/abs.txt ===\nbad\n\n"
               "=== ok.py ===\nprint(1)\n")
        self.assertEqual(list(parse_sections(raw)), ["ok.py"])

    def test_dotdot_in_middle_dropped(self):
        raw = "=== src/../../evil.txt ===\nbad\n"
        self.assertEqual(parse_sections(raw), {})


class TestParseJsonLenient(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(parse_json_lenient('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_truncated_array_salvaged(self):
        raw = '[{"a": 1}, {"b": 2}, {"c": "trunc'
        self.assertEqual(parse_json_lenient(raw), [{"a": 1}, {"b": 2}])


class TestBrokenPythonFiles(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(broken_python_files({"a.py": "x = 1", "b.md": "((("}), [])

    def test_broken(self):
        self.assertEqual(broken_python_files({"a.py": "def f(:\n  pass"}), ["a.py"])


class TestSanitizeLog(unittest.TestCase):
    def test_fabricated_history_stripped(self):
        model_log = "- 2024-07-07: invented entry\n- 2026-07-15: real entry"
        self.assertEqual(sanitize_log("", model_log, "2026-07-15"),
                         "- 2026-07-15: real entry")

    def test_prior_lines_survive_model_mangling(self):
        old = "- 2026-07-01: added caching\n"
        model_log = "- 2026-07-15: new thing\n"  # model dropped the old line
        result = sanitize_log(old, model_log, "2026-07-15")
        self.assertIn("added caching", result)
        self.assertIn("new thing", result)

    def test_no_valid_line_returns_old(self):
        self.assertEqual(sanitize_log("old\n", "- 2024-01-01: wrong date", "2026-07-15"),
                         "old\n")


class TestRotation(unittest.TestCase):
    TODAY = __import__("datetime").date(2026, 7, 19)

    def test_never_matured_first(self):
        reg = [
            {"full_name": "a/1", "created": "2026-07-10", "last_matured": "2026-07-13"},
            {"full_name": "a/2", "created": "2026-07-08", "last_matured": None},
        ]
        self.assertEqual(pick_due_repo(reg, self.TODAY)["full_name"], "a/2")

    def test_oldest_last_matured_next(self):
        reg = [
            {"full_name": "a/1", "created": "2026-07-10", "last_matured": "2026-07-13"},
            {"full_name": "a/2", "created": "2026-07-08", "last_matured": "2026-07-10"},
        ]
        self.assertEqual(pick_due_repo(reg, self.TODAY)["full_name"], "a/2")

    def test_empty(self):
        self.assertIsNone(pick_due_repo([]))

    def test_dormant_repo_skipped_for_active_one(self):
        # a/1 is old + unstarred (dormant) and visited recently -> skipped
        # even though a/2 was visited more recently, because a/2 is starred
        reg = [
            {"full_name": "a/1", "created": "2026-05-01", "stars": 0,
             "last_matured": "2026-07-10"},
            {"full_name": "a/2", "created": "2026-05-01", "stars": 3,
             "last_matured": "2026-07-15"},
        ]
        self.assertEqual(pick_due_repo(reg, self.TODAY)["full_name"], "a/2")

    def test_dormant_repo_due_after_a_month(self):
        reg = [
            {"full_name": "a/1", "created": "2026-05-01", "stars": 0,
             "last_matured": "2026-06-01"},  # 48 days untouched -> due
            {"full_name": "a/2", "created": "2026-05-01", "stars": 3,
             "last_matured": "2026-07-15"},
        ]
        self.assertEqual(pick_due_repo(reg, self.TODAY)["full_name"], "a/1")

    def test_young_unstarred_repo_is_active(self):
        reg = [
            {"full_name": "a/1", "created": "2026-07-18", "stars": 0,
             "last_matured": None},
        ]
        self.assertEqual(pick_due_repo(reg, self.TODAY)["full_name"], "a/1")

    def test_all_dormant_never_stalls(self):
        reg = [
            {"full_name": "a/1", "created": "2026-05-01", "stars": 0,
             "last_matured": "2026-07-18"},
        ]
        self.assertIsNotNone(pick_due_repo(reg, self.TODAY))


class TestCommitSummary(unittest.TestCase):
    def test_new_line_extracted(self):
        old = "- 2026-07-01: a\n"
        new = old + "- 2026-07-15: added vector store\n"
        self.assertEqual(commit_summary(old, new), "2026-07-15: added vector store")


class TestDeriveTopics(unittest.TestCase):
    def test_base_always_present(self):
        for t in derive_topics("completely unmatched words here"):
            self.assertIn(t, ["autoscout", "ai-generated", "agentic-ai"])

    def test_specific_tags(self):
        topics = derive_topics("Challenges in Agent Memory Persistence and State Management")
        self.assertIn("agent-memory", topics)
        self.assertIn("state-management", topics)

    def test_github_topic_format(self):
        for t in derive_topics("agent memory tool calling mcp evaluation deployment"):
            self.assertRegex(t, r"^[a-z0-9][a-z0-9\-]*$")
            self.assertLessEqual(len(t), 50)


class TestNearDuplicate(unittest.TestCase):
    REAL_PAIR_EXISTING = ["LLMs consume excessive tokens before processing prompts"]

    def test_real_world_duplicate_caught(self):
        self.assertIsNotNone(find_near_duplicate(
            "LLMs process excessive tokens before prompt", self.REAL_PAIR_EXISTING))

    def test_distinct_problem_not_flagged(self):
        self.assertIsNone(find_near_duplicate(
            "Agent sandbox escape detection", self.REAL_PAIR_EXISTING))


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello, World! 123"), "hello-world-123")


if __name__ == "__main__":
    unittest.main()
