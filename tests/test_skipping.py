"""Step C tests: turning stages off with "enabled": false (pass-through, notes, validation).

Run from the project folder:
    python3 -m unittest tests.test_skipping -v
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core import schema                            # noqa: E402
from core.config import ConfigError                # noqa: E402
from stages import package                         # noqa: E402
from tests import test_judge, test_package         # noqa: E402
from tests.fakes import FakeLLM                    # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

STORY = "Dana Whitfield spoke to the BBC on Tuesday."


class Plain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        self.out = []

    def off(self, *names):
        for n in names:
            self.cfg["stages"].setdefault(n, {})["enabled"] = False

    def run_stage(self, job, name, **kw):
        with FakeLLM() as fake:                        # no scripted replies: any model call would fail the test
            result = pipeline.run_stage(self.cfg, job, name, out=self.out.append, **kw)
        self.assertEqual(fake.calls, [])
        return result


class TurnedOffStages(Plain):
    def test_simplify_off_passes_the_article_through(self):
        self.off("simplify")
        job = job_with(self.cfg, **{"01_clean.txt": STORY + "\n"})
        self.assertEqual(self.run_stage(job, "simplify"), "turned_off")
        self.assertEqual(job.read("02_simplified.txt").strip(), STORY)
        self.assertEqual(job.status()["meta"]["simplification"], "skipped")
        self.assertEqual(job.status()["stages"]["simplify"]["status"], "skipped")
        self.assertIn("turned off", self.out[0])

    def test_tag_off_passes_the_story_through_with_nothing_protected(self):
        self.off("tag")
        job = job_with(self.cfg, **{"02_simplified.txt": STORY + "\n"})
        self.assertEqual(self.run_stage(job, "tag"), "turned_off")
        self.assertEqual(job.read("03_tagged.txt").strip(), STORY)
        self.assertEqual(job.status()["meta"]["tagging"], "skipped")

    def test_a_turned_off_stage_that_needs_a_missing_file_says_which_stage_makes_it(self):
        self.off("simplify")
        job = job_with(self.cfg)
        with self.assertRaisesRegex(pipeline.StageError, r"simplify needs 01_clean.txt.*run.py ingest --job"):
            self.run_stage(job, "simplify")

    def test_running_it_again_does_not_redo_the_pass_through(self):
        self.off("simplify")
        job = job_with(self.cfg, **{"01_clean.txt": STORY + "\n"})
        self.run_stage(job, "simplify")
        job.write("02_simplified.txt", "edited by hand\n")
        self.assertEqual(self.run_stage(job, "simplify"), "turned_off")
        self.assertEqual(job.read("02_simplified.txt"), "edited by hand\n")

    def test_turning_a_stage_off_forgets_the_later_stages(self):
        self.off("simplify")
        job = job_with(self.cfg, **{"01_clean.txt": STORY + "\n"})
        pipeline._set_status(job, "tag", {"status": "done"})
        self.run_stage(job, "simplify", force=True)
        self.assertNotIn("tag", job.status()["stages"])

    def test_revert_off_removes_old_files_and_turns_the_comprehension_judge_off_too(self):
        self.off("revert")                              # the judge is not switched off by hand
        job = job_with(self.cfg, **{"07_meta.json": "{}", "07_reversion_1.txt": "x", "08_winner.txt": "y"})
        self.assertEqual(self.run_stage(job, "revert"), "turned_off")
        self.assertEqual(self.run_stage(job, "judge_comprehension"), "turned_off")
        self.assertFalse(any(job.exists(f) for f in ("07_meta.json", "07_reversion_1.txt", "08_winner.txt")))

    def test_off_wins_over_a_false_condition_and_a_missing_input(self):
        self.off("revert")
        job = job_with(self.cfg)                        # no 06_count.json at all
        self.assertEqual(self.run_stage(job, "revert"), "turned_off")


class JudgesOff(test_judge.Base):
    def test_judge_translation_off_takes_the_middle_candidate_with_no_call(self):
        self.cfg["stages"]["judge_translation"]["enabled"] = False
        with FakeLLM() as fake:
            result = pipeline.run_stage(self.cfg, self.job, "judge_translation", out=lambda s: None)
        self.assertEqual((result, fake.calls), ("turned_off", []))
        self.assertEqual(self.job.read_json("05_judge.json")["winner"], 2)
        self.assertEqual(self.job.read("05_winner.txt").strip(), test_judge.T[1][0])
        self.assertEqual(self.job.status()["stages"]["judge_translation"]["status"], "skipped")


class PackageNotes(test_package.Base):
    def test_the_notes_list_what_was_turned_off_and_english_says_not_simplified(self):
        job = self.job(test_package.OK_TEXT)
        job.set_meta(simplification="skipped")
        pipeline._set_status(job, "simplify", {"status": "skipped", "note": "turned off in config.json"})
        pipeline._set_status(job, "tag", {"status": "skipped", "note": "turned off in config.json"})
        package.run(self.cfg, job)
        result = job.read_json("09_result.json")
        self.assertIn("Steps turned off in config.json: simplify, tag.", result["notes"])
        self.assertEqual(result["english"], "(not simplified)")
        self.assertEqual(result["warnings"], [])

    def test_revert_off_while_reversion_is_needed_says_so(self):
        job = self.job(test_package.BIG_TEXT)
        pipeline._set_status(job, "revert", {"status": "skipped", "note": "turned off in config.json"})
        package.run(self.cfg, job)
        notes = " ".join(job.read_json("09_result.json")["notes"])
        self.assertIn("reversion would have been used, but it is turned off", notes)
        self.assertNotIn("no attempt passed", notes)


class Validation(Plain):
    def problems(self):
        return schema.validate(self.cfg)

    def test_the_shipped_defaults_are_valid(self):
        self.assertEqual(self.problems(), [])

    def test_always_needed_stages_cannot_be_turned_off(self):
        for name in schema.ALWAYS_ON:
            cfg = copy.deepcopy(self.cfg)
            cfg["stages"].setdefault(name, {})["enabled"] = False
            problems = schema.validate(cfg)
            self.assertTrue(any(f"stages.{name}.enabled" in p and "cannot" in p for p in problems), (name, problems))
            self.assertFalse(any("is not a setting" in p for p in problems))

    def test_enabled_must_be_true_or_false_without_quotes(self):
        self.cfg["stages"]["tag"]["enabled"] = "false"
        self.assertTrue(any("stages.tag.enabled" in p and "true or false" in p for p in self.problems()))

    def test_mail_out_is_off_by_default_the_others_on(self):
        self.assertFalse(schema.is_enabled(self.cfg, "mail_out"))
        for name in ("simplify", "tag", "judge_translation", "revert", "judge_comprehension"):
            self.assertTrue(schema.is_enabled(self.cfg, name))

    def test_the_old_mail_enabled_setting_points_to_its_new_home(self):
        self.cfg["mail"] = {"enabled": True}
        self.assertTrue(any('"mail.enabled"' in p and "stages.mail_out.enabled" in p for p in self.problems()))

    def test_config_lists_turned_off_stages_and_does_not_count_them_as_users_of_a_model(self):
        self.off("simplify", "tag", "translate" if False else "revert")
        lines = "\n".join(schema.describe(self.cfg, {}))
        self.assertIn("simplify             OFF", lines)
        self.assertNotIn("simplify, ", lines.split("Stages:")[0])


if __name__ == "__main__":
    unittest.main()
