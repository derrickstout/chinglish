"""Step 6 tests: stages/count.py (counting new characters and deciding on reversion). No model is used.

Run from the project folder:
    python3 tests/test_count.py -v
    python3 -m unittest discover tests
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core.config import ConfigError                # noqa: E402
from stages import count                           # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

FIX = BASE / "tests" / "fixtures"
ALLOWED = json.loads((FIX / "allowed_chars_sample.json").read_text(encoding="utf-8"))
KNOWN = ALLOWED["mastered"] + ALLOWED["learning"]
POOL = [u["char"] for u in ALLOWED["unknown"]]
# Characters in no list at all (rare ones), for building texts with a known number of new characters.
RARE = [chr(c) for c in range(0x9F00, 0x9F00 + 80) if chr(c) not in KNOWN and chr(c) not in POOL]


def text_with(new, known=5):
    """A translation using exactly `new` different new characters (plus a few known ones)."""
    return "".join(KNOWN[:known]) + "".join(RARE[:new]) + "。"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        (Path(self.tmp.name) / "data").mkdir()
        shutil.copy(FIX / "allowed_chars_sample.json", Path(self.tmp.name) / "data" / "allowed_chars.json")

    def job(self, text, tagged=None, tagging=None):
        files = {"05_winner.txt": text + "\n"}
        if tagged is not None:
            files["03_tagged.txt"] = tagged + "\n"
        job = job_with(self.cfg, **files)
        if tagging:
            job.set_meta(tagging=tagging)
        return job


class Decision(Base):
    def run_count(self, new, **cfg_translation):
        self.cfg["translation"].update(cfg_translation)
        job = self.job(text_with(new))
        note = count.run(self.cfg, job)
        return job.read_json("06_count.json"), note

    def test_within_the_target_needs_no_reversion(self):
        r, note = self.run_count(40)
        self.assertEqual((r["new_chars_count"], r["revert_needed"], r["over_by"]), (40, False, 0))
        self.assertIn("within the target", note)

    def test_exactly_at_the_target_is_within_it(self):
        self.assertFalse(self.run_count(50)[0]["revert_needed"])

    def test_over_the_target_but_within_the_tolerance_needs_no_reversion(self):
        r, note = self.run_count(60)
        self.assertEqual((r["over_by"], r["revert_needed"]), (10, False))
        self.assertIn("within the tolerance", note)

    def test_one_past_the_tolerance_needs_reversion(self):
        r, note = self.run_count(61)
        self.assertTrue(r["revert_needed"])
        self.assertEqual(r["revert_threshold"], 60)
        self.assertIn("11 over the target, so English reversion will run", note)

    def test_the_target_and_tolerance_come_from_config(self):
        r, _ = self.run_count(12, target_new_chars=5, revert_tolerance=3)
        self.assertEqual((r["target"], r["tolerance"], r["revert_needed"]), (5, 3, True))
        r, _ = self.run_count(8, target_new_chars=5, revert_tolerance=3)
        self.assertFalse(r["revert_needed"])

    def test_a_tolerance_of_zero_means_any_excess_triggers_reversion(self):
        self.assertTrue(self.run_count(51, revert_tolerance=0)[0]["revert_needed"])

    def test_a_missing_tolerance_setting_defaults_to_ten(self):
        del self.cfg["translation"]["revert_tolerance"]
        job = self.job(text_with(61))
        count.run(self.cfg, job)
        self.assertEqual(job.read_json("06_count.json")["tolerance"], 10)

    def test_a_bad_target_setting_is_a_plain_error(self):
        self.cfg["translation"]["target_new_chars"] = "lots"
        with self.assertRaisesRegex(ConfigError, "target_new_chars"):
            count.run(self.cfg, self.job(text_with(3)))


class WhatIsCounted(Base):
    def test_known_characters_are_never_counted_as_new(self):
        job = self.job("".join(KNOWN[:30]) + "。")
        count.run(self.cfg, job)
        self.assertEqual(job.read_json("06_count.json")["new_chars_count"], 0)

    def test_a_repeated_new_character_counts_once(self):
        job = self.job(RARE[0] * 9 + "。")
        count.run(self.cfg, job)
        r = job.read_json("06_count.json")
        self.assertEqual((r["new_chars_count"], r["new_chars"][0]["count"]), (1, 9))

    def test_preferred_list_characters_count_as_new_but_are_marked(self):
        job = self.job(POOL[0] + RARE[0] + "。")
        count.run(self.cfg, job)
        flags = {d["char"]: d["in_pool"] for d in job.read_json("06_count.json")["new_chars"]}
        self.assertEqual(flags, {POOL[0]: True, RARE[0]: False})

    def test_each_new_character_has_a_context_snippet(self):
        job = self.job(text_with(2))
        count.run(self.cfg, job)
        for d in job.read_json("06_count.json")["new_chars"]:
            self.assertTrue(d["context"])

    def test_dnt_tagged_english_is_not_counted_and_is_not_a_problem(self):
        text = "<dnt>Dana</dnt>" + "".join(KNOWN[:5]) + "。"
        job = self.job(text, tagged="<dnt>Dana</dnt> went.", tagging="ok")
        count.run(self.cfg, job)
        r = job.read_json("06_count.json")
        self.assertEqual((r["new_chars_count"], r["problems"]), (0, []))

    def test_a_lost_protected_span_is_reported_as_a_problem_in_the_note(self):
        job = self.job("".join(KNOWN[:5]) + "。", tagged="<dnt>Dana</dnt> went.", tagging="ok")
        note = count.run(self.cfg, job)
        self.assertIn("WARNING: protected text was missing", note)

    def test_spans_are_not_checked_when_tagging_was_discarded(self):
        job = self.job("".join(KNOWN[:5]) + "。", tagged="Dana went.", tagging="discarded")
        count.run(self.cfg, job)
        self.assertEqual(job.read_json("06_count.json")["problems"], [])

    def test_the_plain_text_report_is_written(self):
        job = self.job(text_with(3))
        count.run(self.cfg, job)
        report = job.read("06_count.txt")
        self.assertIn("New characters (not in your Mastered or Learning lists):", report)
        self.assertIn("New-character target (50)", report)

    def test_rerunning_replaces_the_old_result(self):
        job = self.job(text_with(61))
        count.run(self.cfg, job)
        job.write("05_winner.txt", text_with(2) + "\n")
        count.run(self.cfg, job)
        self.assertFalse(job.read_json("06_count.json")["revert_needed"])


class Errors(Base):
    def test_a_missing_character_file_says_how_to_make_it(self):
        (Path(self.tmp.name) / "data" / "allowed_chars.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "pleco_import"):
            count.run(self.cfg, self.job(text_with(3)))

    def test_a_missing_winner_file_says_which_stage_writes_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "05_winner.txt not found"):
            count.run(self.cfg, job_with(self.cfg))


class ThroughThePipeline(Base):
    def go(self, new):
        job = self.job(text_with(new))
        job.mark_stage("judge_translation")
        pipeline.run_stage(self.cfg, job, "count", out=lambda s: None)
        return job

    def test_count_is_marked_done_and_makes_the_file_revert_waits_for(self):
        job = self.go(70)
        self.assertTrue(job.stage_done("count") and job.exists("06_count.json"))
        self.assertTrue(pipeline.reversion_needed(job))

    def test_reversion_is_not_needed_for_a_short_translation(self):
        self.assertFalse(pipeline.reversion_needed(self.go(10)))

    def test_rerunning_the_judge_forgets_the_count(self):
        job = self.go(10)
        pipeline._invalidate_after(job, "judge_translation")
        self.assertFalse(job.stage_done("count"))


if __name__ == "__main__":
    unittest.main()
