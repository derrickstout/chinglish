"""Step 4 tests: stages/translate.py, with a scripted fake model (no LM Studio needed).

Run from the project folder:
    python3 tests/test_translate.py -v
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
from stages import translate                       # noqa: E402
from tests.fakes import FakeLLM, http_error        # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

FIX = BASE / "tests" / "fixtures"
TEMPLATE = (BASE / "prompts" / "translate.txt").read_text(encoding="utf-8")

TAGGED = "<dnt>Dana</dnt> went to the market. She bought two apples."
GOOD = "<dnt>Dana</dnt>去了市場。她買了兩個蘋果。"
OTHER = "<dnt>Dana</dnt>去市場。她買兩個蘋果。"
THIRD = "<dnt>Dana</dnt>到市場去了。她買了二個蘋果。"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        (Path(self.tmp.name) / "data").mkdir()
        shutil.copy(FIX / "allowed_chars_sample.json", Path(self.tmp.name) / "data" / "allowed_chars.json")
        (Path(self.cfg["paths"]["prompts"]) / "translate.txt").write_text(TEMPLATE, encoding="utf-8")
        self.job = job_with(self.cfg, **{"03_tagged.txt": TAGGED + "\n"})
        self.job.set_meta(tagging="ok")

    def run_stage(self, *replies, job=None):
        with FakeLLM(*replies) as fake:
            note = translate.run(self.cfg, job or self.job)
        return note, fake


class TranslateStage(Base):
    def test_makes_three_translations_and_a_meta_file(self):
        note, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertEqual(len(fake.calls), 3)
        for n, text in enumerate((GOOD, OTHER, THIRD), 1):
            self.assertEqual(self.job.read(f"04_translation_{n}.txt"), text + "\n")
        meta = self.job.read_json("04_meta.json")
        self.assertEqual(meta["temperatures"], [0.3, 0.6, 0.9])
        self.assertEqual([c["n"] for c in meta["candidates"]], [1, 2, 3])
        self.assertTrue(note.startswith("3 of 3 translations"))

    def test_the_three_calls_use_the_configured_temperatures(self):
        self.cfg["stages"]["translate"].update(passes=2, temperature=0.2, pass_temp_step=0.3)
        _, fake = self.run_stage(GOOD, OTHER)
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.2, 0.5])

    def test_the_default_temperatures_are_used_when_none_are_configured(self):
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.3, 0.6, 0.9])

    def test_the_prompt_holds_the_story_the_target_and_the_character_lists(self):
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        content = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(len(fake.calls[0]["body"]["messages"]), 1)
        self.assertIn(TAGGED, content)
        self.assertIn("no more than 50 different characters", content)
        self.assertNotIn("{{", content)
        allowed = json.loads((Path(self.tmp.name) / "data" / "allowed_chars.json").read_text(encoding="utf-8"))
        self.assertIn(allowed["mastered"][0], content)
        self.assertIn(allowed["unknown"][0]["char"], content)

    def test_the_same_prompt_is_sent_every_time(self):
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        sent = {c["body"]["messages"][0]["content"] for c in fake.calls}
        self.assertEqual(len(sent), 1)

    def test_dnt_tags_are_kept_in_the_saved_translation(self):
        self.run_stage(GOOD, OTHER, THIRD)
        self.assertIn("<dnt>Dana</dnt>", self.job.read("04_translation_1.txt"))

    def test_a_notes_line_is_split_off_and_recorded(self):
        self.run_stage(GOOD + "\nNOTES: 蘋果 means apple.", OTHER, THIRD)
        self.assertEqual(self.job.read("04_translation_1.txt"), GOOD + "\n")
        self.assertEqual(self.job.read_json("04_meta.json")["candidates"][0]["notes"], "蘋果 means apple.")

    def test_a_code_fence_is_removed(self):
        self.run_stage("```\n" + GOOD + "\n```", OTHER, THIRD)
        self.assertEqual(self.job.read("04_translation_1.txt"), GOOD + "\n")

    def test_new_characters_are_counted_per_translation(self):
        self.run_stage(GOOD, OTHER, THIRD)
        for c in self.job.read_json("04_meta.json")["candidates"]:
            self.assertIsInstance(c["new_chars"], int)
            self.assertIn("problems", c)

    def test_a_translation_that_loses_a_protected_name_is_flagged_not_stopped(self):
        note, _ = self.run_stage("丹娜去了市場。", OTHER, THIRD)
        cand = self.job.read_json("04_meta.json")["candidates"][0]
        self.assertTrue(cand["problems"])
        self.assertIn("WARNING: problems in #1", note)
        self.assertTrue(self.job.exists("04_translation_1.txt"))

    def test_spans_are_not_checked_when_tagging_was_discarded(self):
        self.job.set_meta(tagging="discarded")
        note, _ = self.run_stage("丹娜去了市場。", OTHER, THIRD)
        self.assertNotIn("protected text", json.dumps(self.job.read_json("04_meta.json")))

    def test_one_failed_call_keeps_the_other_two(self):
        note, _ = self.run_stage(GOOD, http_error(500, "crash"), THIRD)
        meta = self.job.read_json("04_meta.json")
        self.assertIn("returned an error", meta["candidates"][1]["error"])
        self.assertFalse(self.job.exists("04_translation_2.txt"))
        self.assertTrue(self.job.exists("04_translation_3.txt"))
        self.assertIn("2 of 3", note)
        self.assertIn("#2 t=0.6: FAILED", note)
        self.assertIn("failed", self.job.read("log.txt"))

    def test_an_empty_reply_counts_as_a_failed_call(self):
        self.run_stage(GOOD, "NOTES: nothing", THIRD)
        self.assertIsNotNone(self.job.read_json("04_meta.json")["candidates"][1]["error"])

    def test_every_call_failing_stops_the_stage_with_the_last_error(self):
        with self.assertRaisesRegex(RuntimeError, "All 3 translation attempts failed"):
            self.run_stage(http_error(500, "a"), http_error(500, "b"), http_error(500, "c"))
        self.assertTrue(self.job.exists("04_meta.json"))      # kept, so you can see what happened

    def test_a_rerun_removes_translations_left_by_the_earlier_run(self):
        self.run_stage(GOOD, OTHER, THIRD)
        self.cfg["stages"]["translate"]["passes"] = 1
        self.run_stage(GOOD)
        self.assertFalse(self.job.exists("04_translation_2.txt"))

    def test_asks_for_a_long_enough_reply(self):
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertGreaterEqual(fake.calls[0]["body"]["max_tokens"], 4096)

    def test_the_prompt_size_estimate_is_logged(self):
        self.run_stage(GOOD, OTHER, THIRD)
        self.assertRegex(self.job.read("log.txt"), r"prompt is about [\d,]+ tokens")


class Errors(Base):
    def test_a_missing_prompt_file_is_a_plain_error_and_no_call_is_made(self):
        (Path(self.cfg["paths"]["prompts"]) / "translate.txt").unlink()
        with FakeLLM() as fake:
            with self.assertRaisesRegex(ConfigError, r"translate\.txt"):
                translate.run(self.cfg, self.job)
        self.assertEqual(fake.calls, [])

    def test_a_prompt_that_lost_a_placeholder_is_an_error(self):
        (Path(self.cfg["paths"]["prompts"]) / "translate.txt").write_text("Translate this.", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "missing .*STORY"):
            self.run_stage()

    def test_a_missing_character_file_says_how_to_make_it(self):
        (Path(self.tmp.name) / "data" / "allowed_chars.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "pleco_import"):
            self.run_stage()

    def test_bad_pass_settings_are_errors(self):
        for key, bad in (("passes", 0), ("passes", "three"), ("temperature", 5), ("pass_temp_step", -1)):
            with self.subTest(key=key, bad=bad):
                self.cfg["stages"]["translate"][key] = bad
                with self.assertRaisesRegex(ConfigError, f"stages.translate.{key}"):
                    self.run_stage()
                self.cfg["stages"]["translate"].update(passes=3, temperature=0.3, pass_temp_step=0.3)

    def test_the_temperature_never_goes_above_max_temperature(self):
        self.cfg["stages"]["translate"].update(passes=3, temperature=0.8, pass_temp_step=0.3, max_temperature=1.0)
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.8, 1.0, 1.0])

    def test_the_stage_uses_the_profile_it_names(self):
        self.cfg["stages"]["translate"]["model"] = "local2"
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertEqual({c["body"]["model"] for c in fake.calls}, {"fake-model-2"})

    def test_max_tokens_override_replaces_the_budget(self):
        self.cfg["stages"]["translate"]["max_tokens_override"] = 777
        _, fake = self.run_stage(GOOD, OTHER, THIRD)
        self.assertEqual({c["body"]["max_tokens"] for c in fake.calls}, {777})

    def test_a_missing_input_file_says_which_stage_writes_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "03_tagged.txt not found"):
            self.run_stage(job=job_with(self.cfg))

    def test_a_story_containing_a_placeholder_is_not_expanded(self):
        job = job_with(self.cfg, **{"03_tagged.txt": "Type {{KNOWN_CHARS}} here.\n"})
        _, fake = self.run_stage(GOOD, OTHER, THIRD, job=job)
        self.assertIn("Type {{KNOWN_CHARS}} here.", fake.calls[0]["body"]["messages"][0]["content"])


class SettingsErrors(Base):
    def test_a_settings_problem_stops_at_once_instead_of_failing_three_times(self):
        self.cfg["models"]["local1"]["model"] = ""
        with FakeLLM() as fake:
            with self.assertRaises(ConfigError):
                translate.run(self.cfg, self.job)
        self.assertEqual(fake.calls, [])


class Helpers(unittest.TestCase):
    def test_preferred_characters_exclude_known_ones(self):
        known, preferred = translate.char_lists(
            {"mastered": ["我"], "learning": ["你"], "unknown": [{"char": "你"}, {"char": "他"}]})
        self.assertEqual((known, preferred), ("我你", "他"))

    def test_the_token_estimate_counts_chinese_heavier_than_english(self):
        self.assertGreater(translate.estimate_tokens("你好世界"), translate.estimate_tokens("abcd"))

    def test_the_shipped_prompt_has_all_four_placeholders(self):
        for p in translate.PLACEHOLDERS:
            self.assertIn(p, TEMPLATE)


class ThroughThePipeline(Base):
    def test_translate_runs_after_tag_is_done_and_marks_itself_done(self):
        self.job.mark_stage("tag")
        out = []
        with FakeLLM(GOOD, OTHER, THIRD):
            result = pipeline.run_stage(self.cfg, self.job, "translate", out=out.append)
        self.assertEqual(result, "done")
        self.assertTrue(self.job.stage_done("translate"))
        self.assertTrue(any("translate: done" in line for line in out))

    def test_rerunning_tag_forgets_the_translation(self):
        self.job.mark_stage("tag")
        with FakeLLM(GOOD, OTHER, THIRD):
            pipeline.run_stage(self.cfg, self.job, "translate", out=lambda s: None)
        self.job.write("02_simplified.txt", "Dana went to the market. She bought two apples.\n")
        with FakeLLM(TAGGED):
            pipeline.run_stage(self.cfg, self.job, "tag", force=True, out=lambda s: None)
        self.assertFalse(self.job.stage_done("translate"))


if __name__ == "__main__":
    unittest.main()
