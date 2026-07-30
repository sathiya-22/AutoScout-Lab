#!/usr/bin/env python3
"""Unit tests for AutoScout-Lab's pipeline logic.

Mostly pure-logic — no network, no Gemini, no filesystem side effects —
except TestVerifyPythonRepo, which really does spin up a venv and run
generated code in a sandbox (that's the point of the module under test).
Run: python3 -m unittest discover tests -v
"""

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import mature_repo  # noqa: E402 — imported as a module so call_gemini can be patched
from eval_harness import (append_score_log, freeze_eval_files,  # noqa: E402
                          is_regression, judge_score, parse_harness_proposal,
                          run_eval)
from generate_prototype import derive_topics  # noqa: E402
from mature_repo import (build_prompt, commit_summary,  # noqa: E402
                         pick_due_repo, sanitize_log)
from research import (build_propose_prompt, extract_result,  # noqa: E402
                      parse_research_proposal, run_research_stage,
                      sanitize_interpretation)
from scout_common import (broken_python_files, parse_json_lenient,  # noqa: E402
                          parse_sections, slugify)
from scout_problems import find_near_duplicate  # noqa: E402
from verify import _classify_failure, verify_python_repo  # noqa: E402


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


class TestVerifyPythonRepo(unittest.TestCase):
    """Real sandboxed execution — the whole point of this module is to
    catch bugs static analysis (the syntax gate) can't see."""

    def test_no_entrypoint_skips(self):
        self.assertTrue(verify_python_repo({"README.md": "hi"})["ok"])

    def test_clean_exit_passes(self):
        self.assertTrue(verify_python_repo({"main.py": "print('hi')\n"})["ok"])

    def test_name_error_flagged(self):
        result = verify_python_repo({"main.py": "print(undefined_var)\n"})
        self.assertFalse(result["ok"])
        self.assertIn("NameError", result["reason"])

    def test_missing_dependency_flagged(self):
        result = verify_python_repo({"main.py": "import nonexistent_pkg_xyz\n"})
        self.assertFalse(result["ok"])

    def test_auth_failure_with_dummy_key_passes(self):
        code = ("import os, sys\n"
               "if os.environ.get('GEMINI_API_KEY') == 'dummy-key-for-verification':\n"
               "    print('401 Unauthorized', file=sys.stderr); sys.exit(1)\n")
        self.assertTrue(verify_python_repo({"main.py": code})["ok"])

    def test_interactive_input_eof_passes(self):
        self.assertTrue(verify_python_repo({"main.py": "input('prompt> ')\n"})["ok"])

class TestClassifyFailureCliUsage(unittest.TestCase):
    def test_rich_boxed_missing_command_is_benign(self):
        # A Typer/Click CLI with >1 @app.command() exits 2 with a rich-styled
        # box when run with zero args — the exact failure mode that caused a
        # real maturation run to wrongly abort (see 2026-07-29 incident): the
        # last line is just a box border, not a recognizable "SomeError:".
        stderr = (
            "Usage: main.py [OPTIONS] COMMAND [ARGS]...\n"
            "Try 'main.py --help' for help.\n"
            "╭─ Error ────────────────────────────────────────────\n"
            "│ Missing command. │\n"
            "╰────────────────────────────────────────────╯\n"
        )
        ok, reason = _classify_failure(stderr)
        self.assertTrue(ok)
        self.assertIn("subcommand", reason)

    def test_real_structural_bug_still_flagged(self):
        stderr = "Traceback (most recent call last):\nNameError: name 'x' is not defined\n"
        ok, reason = _classify_failure(stderr)
        self.assertFalse(ok)


class TestVerifyWithRetries(unittest.TestCase):
    def test_succeeds_first_try_no_model_call_needed(self):
        verified, _ = mature_repo.verify_with_retries(
            "fake-key", {}, {"main.py": "print('hi')\n"})
        self.assertIsNotNone(verified)

    def test_fix_applied_on_retry(self):
        broken = {"main.py": "print(undefined_var)\n"}
        fixed_raw = "=== main.py ===\nprint('fixed')\n"
        with unittest.mock.patch("mature_repo.call_gemini", return_value=fixed_raw):
            verified, _ = mature_repo.verify_with_retries("fake-key", {}, broken)
        self.assertIsNotNone(verified)
        self.assertEqual(verified["main.py"], "print('fixed')")

    def test_gives_up_after_retries_exhausted(self):
        broken = {"main.py": "print(undefined_var)\n"}
        still_broken_raw = "=== main.py ===\nprint(undefined_var)\n"
        with unittest.mock.patch("mature_repo.call_gemini", return_value=still_broken_raw):
            verified, _ = mature_repo.verify_with_retries("fake-key", {}, broken)
        self.assertIsNone(verified)

    def test_untouched_dependent_file_break_is_caught(self):
        # main.py isn't edited this pass, but the edited config.py breaks it —
        # verifying edited_files alone would miss this entirely. Mock the
        # fix attempts (never fixes it) so this doesn't need google-genai.
        base = {"main.py": "from config import VALUE\nprint(VALUE)\n"}
        edited = {"config.py": "# VALUE removed by mistake\n"}
        still_broken_raw = "=== config.py ===\n# still broken\n"
        with unittest.mock.patch("mature_repo.call_gemini", return_value=still_broken_raw):
            verified, reason = mature_repo.verify_with_retries("fake-key", base, edited)
        self.assertIsNone(verified)
        self.assertIn("ImportError", reason)


class TestResearchProposalParsing(unittest.TestCase):
    def test_benchmark_proposal_parsed(self):
        raw = ("QUESTION: is list comprehension faster than a for-loop here?\n"
              "TYPE: benchmark\n"
              "=== research/loop_bench.py ===\n"
              "print('AUTOSCOUT_RESEARCH_RESULT: {\"a\": 1}')\n")
        parsed = parse_research_proposal(raw)
        self.assertEqual(parsed["type"], "benchmark")
        self.assertEqual(parsed["script_filename"], "research/loop_bench.py")
        self.assertIn("AUTOSCOUT_RESEARCH_RESULT", parsed["script_content"])

    def test_qualitative_proposal_parsed(self):
        raw = "QUESTION: which library fits better?\nTYPE: qualitative\nNOTES: library X has more features.\n"
        parsed = parse_research_proposal(raw)
        self.assertEqual(parsed["type"], "qualitative")
        self.assertIn("library X", parsed["notes"])

    def test_missing_question_rejected(self):
        self.assertIsNone(parse_research_proposal("TYPE: benchmark\n"))

    def test_path_traversal_rejected(self):
        raw = ("QUESTION: q\nTYPE: benchmark\n"
              "=== research/../../evil.py ===\nprint('hi')\n")
        self.assertIsNone(parse_research_proposal(raw))


class TestExtractResult(unittest.TestCase):
    def test_extracts_json_after_marker(self):
        stdout = "some noise\nAUTOSCOUT_RESEARCH_RESULT: {\"ms\": 12.5}\nmore noise\n"
        self.assertEqual(extract_result(stdout), {"ms": 12.5})

    def test_no_marker_returns_none(self):
        self.assertIsNone(extract_result("just some output\n"))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(extract_result("AUTOSCOUT_RESEARCH_RESULT: not json\n"))


class TestSanitizeInterpretation(unittest.TestCase):
    def test_clean_interpretation_kept(self):
        result = {"a_ms": 12.5, "b_ms": 31.0}
        text = "Approach A at 12.5ms beats approach B at 31.0ms — adopt A."
        self.assertEqual(sanitize_interpretation(text, result, "q"), text)

    def test_fabricated_number_discarded(self):
        result = {"a_ms": 12.5, "b_ms": 31.0}
        text = "This is a 47% improvement, clearly worth adopting."
        out = sanitize_interpretation(text, result, "q")
        self.assertIn("discarded", out)

    def test_number_from_question_allowed(self):
        result = {"speedup": 2.0}
        text = "Comparing across 5 runs confirms a speedup of 2.0x."
        # "5" appears in the question, "2.0" appears in the result — nothing foreign.
        self.assertEqual(
            sanitize_interpretation(text, result, "average of 5 runs"), text)


class TestRunResearchStage(unittest.TestCase):
    def test_benchmark_success_produces_entry_and_script_file(self):
        proposal_raw = ("QUESTION: is A faster than B?\nTYPE: benchmark\n"
                        "=== research/bench.py ===\n"
                        "print('AUTOSCOUT_RESEARCH_RESULT: {\"a_ms\": 1.0, \"b_ms\": 2.0}')\n")
        interp_raw = "A at 1.0ms beats B at 2.0ms — adopt A."
        calls = iter([proposal_raw, interp_raw])
        call_llm = unittest.mock.Mock(side_effect=lambda *a, **k: next(calls))

        extra_files, entry_text = run_research_stage(
            call_llm, "fake-key", {"full_name": "x/y", "name": "y", "topic": "t", "iterations": 0},
            {"main.py": "print('hi')\n"}, "", "2026-07-27", 1)

        self.assertIn("research/bench.py", extra_files)
        self.assertIn("Measured result", entry_text)
        self.assertIn("adopt A", entry_text)

    def test_no_proposal_yields_nothing(self):
        call_llm = unittest.mock.Mock(return_value="garbage, no fields")
        extra_files, entry_text = run_research_stage(
            call_llm, "fake-key", {"full_name": "x/y", "name": "y", "topic": "t", "iterations": 0},
            {"main.py": "print('hi')\n"}, "", "2026-07-27", 1)
        self.assertEqual(extra_files, {})
        self.assertEqual(entry_text, "")

    def test_qualitative_proposal_logged_as_unverified(self):
        call_llm = unittest.mock.Mock(
            return_value="QUESTION: which lib?\nTYPE: qualitative\nNOTES: lib X seems richer.\n")
        extra_files, entry_text = run_research_stage(
            call_llm, "fake-key", {"full_name": "x/y", "name": "y", "topic": "t", "iterations": 0},
            {"main.py": "print('hi')\n"}, "", "2026-07-27", 1)
        self.assertEqual(extra_files, {})
        self.assertIn("unverified", entry_text)


class TestEvalHarnessParsing(unittest.TestCase):
    def test_valid_proposal_parsed(self):
        raw = ("=== eval/dataset.json ===\n[{\"input\": 1}]\n"
              "=== eval/run_eval.py ===\nprint('AUTOSCOUT_EVAL_SCORE: {\"score\": 1.0}')\n")
        parsed = parse_harness_proposal(raw)
        self.assertIn("eval/dataset.json", parsed)
        self.assertIn("eval/run_eval.py", parsed)

    def test_missing_script_rejected(self):
        raw = "=== eval/dataset.json ===\n[]\n"
        self.assertIsNone(parse_harness_proposal(raw))

    def test_path_traversal_rejected(self):
        raw = ("=== eval/dataset.json ===\n[]\n"
              "=== eval/../../evil.py ===\nprint('hi')\n")
        parsed = parse_harness_proposal(raw)
        self.assertIsNone(parsed)


class TestFreezeEvalFiles(unittest.TestCase):
    def test_pre_existing_eval_files_stripped(self):
        original = {"eval/dataset.json": "[]", "eval/run_eval.py": "old"}
        edited = {"eval/run_eval.py": "model tried to rewrite this",
                 "main.py": "print(1)"}
        result = freeze_eval_files(edited, original)
        self.assertNotIn("eval/run_eval.py", result)
        self.assertIn("main.py", result)

    def test_newly_created_eval_files_not_stripped(self):
        # not in `original` (didn't exist before this cycle) — allowed through
        edited = {"eval/run_eval.py": "brand new"}
        result = freeze_eval_files(edited, {})
        self.assertIn("eval/run_eval.py", result)


class TestIsRegression(unittest.TestCase):
    def test_lower_after_is_regression(self):
        self.assertTrue(is_regression({"score": 5.0}, {"score": 3.0}))

    def test_equal_or_higher_is_not_regression(self):
        self.assertFalse(is_regression({"score": 5.0}, {"score": 5.0}))
        self.assertFalse(is_regression({"score": 5.0}, {"score": 9.0}))

    def test_missing_score_is_inconclusive_not_regression(self):
        self.assertFalse(is_regression(None, {"score": 1.0}))
        self.assertFalse(is_regression({"score": 5.0}, None))


class TestJudgeScore(unittest.TestCase):
    def test_parses_json_from_model_response(self):
        call_llm = unittest.mock.Mock(
            return_value='{"quality_score": 8, "reasoning": "solid improvement"}')
        result = judge_score(call_llm, "fake-key", "diff", {"score": 1}, {"score": 2})
        self.assertEqual(result["quality_score"], 8)

    def test_unparseable_response_returns_none(self):
        call_llm = unittest.mock.Mock(return_value="not json at all")
        self.assertIsNone(judge_score(call_llm, "fake-key", "diff", {"score": 1}, {"score": 2}))


class TestAppendScoreLog(unittest.TestCase):
    def test_appends_valid_json_line(self):
        log = append_score_log("", "2026-07-27", 1, {"score": 3.0}, {"quality_score": 7})
        entry = json.loads(log.strip())
        self.assertEqual(entry["iteration"], 1)
        self.assertEqual(entry["score"]["score"], 3.0)


class TestRunEvalNoScript(unittest.TestCase):
    def test_missing_script_returns_none(self):
        self.assertIsNone(run_eval({"main.py": "print(1)"}))


class TestPromptDumpExcludesArtifacts(unittest.TestCase):
    """Real incident (2026-07-29/30, AutoScout-Engine): research/eval
    artifact files riding along in the main prompt's file dump pushed a
    long-lived repo's advancement call over Groq's 12,000 TPM limit (413
    Request too large). The fix applies here too, ahead of Gemini's much
    larger budget ever hitting the same wall."""

    entry = {"full_name": "x/y", "name": "y", "topic": "t", "iterations": 0}

    def test_research_and_eval_files_excluded_from_dump(self):
        files = {
            "main.py": "print('core')",
            "research/bench_1.py": "print('AUTOSCOUT_RESEARCH_RESULT: {}')",
            "eval/dataset.json": "[]",
            "eval/run_eval.py": "print('AUTOSCOUT_EVAL_SCORE: {}')",
        }
        prompt = build_prompt(self.entry, files)
        self.assertIn("main.py", prompt)
        self.assertNotIn("bench_1.py", prompt)
        self.assertNotIn("run_eval.py", prompt)

    def test_research_propose_prompt_also_excludes_artifacts(self):
        files = {"main.py": "print('core')", "eval/run_eval.py": "print('x')"}
        prompt = build_propose_prompt(self.entry, files, "")
        self.assertIn("main.py", prompt)
        self.assertNotIn("run_eval.py", prompt)


if __name__ == "__main__":
    unittest.main()
