"""Step 7 tests: core/tags.py, stages/revert.py and judge.run_comprehension, with scripted fake models.

Run from the project folder:
    python3 tests/test_revert.py -v
    python3 -m unittest discover tests
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core import tags                              # noqa: E402
from core.config import ConfigError                # noqa: E402
from stages import count, judge, revert            # noqa: E402
from tests.fakes import FakeLLM, http_error        # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

FIX = BASE / "tests" / "fixtures"
ALLOWED = json.loads((FIX / "allowed_chars_sample.json").read_text(encoding="utf-8"))
KNOWN = ALLOWED["mastered"] + ALLOWED["learning"]
POOL = [u["char"] for u in ALLOWED["unknown"]]
RARE = [chr(c) for c in range(0x9F00, 0x9F00 + 120) if chr(c) not in KNOWN and chr(c) not in POOL]

ENGLISH = "<dnt>Dana</dnt> walked to the market and bought apples near the river."
WINNER = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:70]) + "。"
# Reversions: each puts the English back for some of the new characters.
GOOD_1 = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:30]) + "<en>market</en>" + "。"
GOOD_2 = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:20]) + "<en>apples</en>和<en>river</en>。"
GOOD_3 = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:40]) + "<en>market</en>。"


class Tags(unittest.TestCase):
    def test_strip_and_extract(self):
        t = "他<en>market</en>和<dnt>Dana</dnt>"
        self.assertEqual(tags.strip_en_tags(t), "他market和<dnt>Dana</dnt>")
        self.assertEqual(tags.extract_en(t), ["market"])
        self.assertEqual(tags.strip_all_tags(t), "他market和Dana")

    def test_balance(self):
        self.assertTrue(tags.balanced_en("a<en>b</en>c<en>d</en>"))
        self.assertTrue(tags.balanced_en("no tags"))
        for bad in ("<en>a", "a</en>", "<en>a<en>b</en></en>"):
            with self.subTest(bad=bad):
                self.assertFalse(tags.balanced_en(bad))

    def test_drop_en_text_removes_english_but_keeps_a_dnt_inside(self):
        self.assertEqual(tags.drop_en_text("他<en>big market</en>去").replace("  ", " "), "他 去")
        self.assertIn("<dnt>Dana</dnt>", tags.drop_en_text("<en>with <dnt>Dana</dnt></en>"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        root = Path(self.tmp.name)
        (root / "data").mkdir()
        shutil.copy(FIX / "allowed_chars_sample.json", root / "data" / "allowed_chars.json")
        prompts = Path(self.cfg["paths"]["prompts"])
        for name in ("revert.txt", "judge_comprehension.txt", "judge_comprehension_rubric.txt"):
            (prompts / name).write_text((BASE / "prompts" / name).read_text(encoding="utf-8"), encoding="utf-8")
        env = mock.patch.dict(os.environ, {"TEST_CLOUD_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)
        self.job = job_with(self.cfg, **{"03_tagged.txt": ENGLISH + "\n", "05_winner.txt": WINNER + "\n"})
        self.job.set_meta(tagging="ok")
        count.run(self.cfg, self.job)                      # makes the real 06_count.json

    def run_revert(self, *replies, job=None):
        with FakeLLM(*replies) as fake:
            note = revert.run(self.cfg, job or self.job)
        return note, fake


class RevertStage(Base):
    def test_the_count_stage_sets_up_the_scenario(self):
        r = self.job.read_json("06_count.json")
        self.assertEqual((r["new_chars_count"], r["revert_needed"]), (70, True))

    def test_three_valid_attempts_are_saved_and_recorded(self):
        note, fake = self.run_revert(GOOD_1, GOOD_2, GOOD_3)
        self.assertEqual(len(fake.calls), 3)
        for n, text in enumerate((GOOD_1, GOOD_2, GOOD_3), 1):
            self.assertEqual(self.job.read(f"07_reversion_{n}.txt"), text + "\n")
        meta = self.job.read_json("07_meta.json")
        self.assertEqual([c["valid"] for c in meta["candidates"]], [True, True, True])
        self.assertEqual([c["new_chars"] for c in meta["candidates"]], [30, 20, 40])
        self.assertEqual([c["en_spans"] for c in meta["candidates"]], [1, 2, 1])
        self.assertEqual(meta["winner_new_chars"], 70)
        self.assertTrue(note.startswith("3 of 3 valid (winner had 70)"))

    def test_the_prompt_holds_the_english_the_translation_and_the_new_characters(self):
        _, fake = self.run_revert(GOOD_1, GOOD_2, GOOD_3)
        content = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(len(fake.calls[0]["body"]["messages"]), 1)
        self.assertIn(ENGLISH, content)
        self.assertIn(WINNER, content)
        self.assertIn(RARE[0], content)
        self.assertIn("not on the preferred list", content)
        self.assertIn("no more than 50 different new characters", content)
        self.assertNotIn("{{", content)

    def test_the_same_prompt_is_sent_at_the_configured_temperatures(self):
        self.cfg["stages"]["revert"].update(passes=2, temperature=0.2, pass_temp_step=0.3)
        _, fake = self.run_revert(GOOD_1, GOOD_2)
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.2, 0.5])
        self.assertEqual(len({c["body"]["messages"][0]["content"] for c in fake.calls}), 1)

    def test_default_temperatures(self):
        self.cfg["stages"]["revert"] = {"model": "local1"}            # everything else at its default
        _, fake = self.run_revert(GOOD_1, GOOD_2, GOOD_3)
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.3, 0.6, 0.9])

    def test_a_fence_an_introduction_and_a_notes_line_are_removed(self):
        self.run_revert("```\nHere is the edited text:\n" + GOOD_1 + "\nNOTES: swapped market\n```", GOOD_2, GOOD_3)
        self.assertEqual(self.job.read("07_reversion_1.txt"), GOOD_1 + "\n")
        self.assertEqual(self.job.read_json("07_meta.json")["candidates"][0]["notes"], "swapped market")


class Verification(Base):
    def attempt(self, bad):
        self.run_revert(bad, GOOD_2, GOOD_3)
        c = self.job.read_json("07_meta.json")["candidates"][0]
        self.assertTrue(self.job.exists("07_reversion_1.txt"))          # rejected attempts are kept for you to read
        return c

    def test_unbalanced_en_tags_are_rejected(self):
        c = self.attempt(GOOD_1.replace("</en>", ""))
        self.assertFalse(c["valid"])
        self.assertIn("the <en> tags were unbalanced", c["problems"])

    def test_english_that_is_not_in_the_original_is_rejected(self):
        c = self.attempt(GOOD_1.replace("<en>market</en>", "<en>supermarket</en>").replace("<en>", "<en>").replace(
            "supermarket", "bazaar"))
        self.assertFalse(c["valid"])
        self.assertTrue(any("not in the original" in p and "bazaar" in p for p in c["problems"]))

    def test_english_matching_ignores_case_and_spacing(self):
        c = self.attempt(GOOD_1.replace("<en>market</en>", "<en>  The   Market </en>"))
        self.assertTrue(c["valid"])

    def test_an_empty_en_span_is_rejected(self):
        self.assertFalse(self.attempt(GOOD_1.replace("<en>market</en>", "<en></en>"))["valid"])

    def test_a_lost_protected_name_is_rejected(self):
        c = self.attempt(GOOD_1.replace("<dnt>Dana</dnt>", ""))
        self.assertFalse(c["valid"])
        self.assertTrue(any("protected text" in p for p in c["problems"]))

    def test_a_name_inside_an_en_span_still_counts_as_kept(self):
        text = GOOD_1.replace("<dnt>Dana</dnt>", "<en><dnt>Dana</dnt> walked</en>")
        self.assertTrue(self.attempt(text)["valid"])

    def test_an_attempt_that_does_not_reduce_the_count_is_rejected(self):
        c = self.attempt(WINNER)
        self.assertFalse(c["valid"])
        self.assertTrue(any("did not drop" in p for p in c["problems"]))

    def test_an_attempt_that_adds_new_characters_is_rejected(self):
        self.assertFalse(self.attempt(WINNER + RARE[90])["valid"])

    def test_a_reply_with_no_chinese_is_rejected(self):
        c = self.attempt("<en>Dana walked to the market</en>")
        self.assertFalse(c["valid"])

    def test_spans_are_not_checked_when_tagging_was_discarded(self):
        self.job.set_meta(tagging="discarded")
        self.assertTrue(self.attempt(GOOD_1.replace("<dnt>Dana</dnt>", ""))["valid"])

    def test_the_note_warns_when_nothing_is_valid(self):
        note, _ = self.run_revert(WINNER, WINNER, WINNER)
        self.assertIn("0 of 3 valid", note)
        self.assertIn("WARNING: none passed the checks", note)
        self.assertTrue(self.job.exists("07_meta.json"))

    def test_a_failed_call_keeps_the_others(self):
        note, _ = self.run_revert(GOOD_1, http_error(500, "boom"), GOOD_3)
        meta = self.job.read_json("07_meta.json")
        self.assertIn("returned an error", meta["candidates"][1]["error"])
        self.assertFalse(self.job.exists("07_reversion_2.txt"))
        self.assertIn("#2 t=0.6: FAILED", note)

    def test_every_call_failing_stops_the_stage(self):
        with self.assertRaisesRegex(RuntimeError, "All 3 reversion attempts failed"):
            self.run_revert(http_error(500, "a"), http_error(500, "b"), http_error(500, "c"))


class RevertErrors(Base):
    def test_a_missing_prompt_is_a_plain_error_with_no_call(self):
        (Path(self.cfg["paths"]["prompts"]) / "revert.txt").unlink()
        with FakeLLM() as fake:
            with self.assertRaisesRegex(ConfigError, r"revert\.txt"):
                revert.run(self.cfg, self.job)
        self.assertEqual(fake.calls, [])

    def test_a_prompt_that_lost_a_placeholder_is_an_error(self):
        (Path(self.cfg["paths"]["prompts"]) / "revert.txt").write_text("Fix it. {{ENGLISH}}", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "missing .*TRANSLATION"):
            self.run_revert()

    def test_a_bad_pass_setting_is_an_error(self):
        self.cfg["stages"]["revert"]["passes"] = "hot"
        with self.assertRaisesRegex(ConfigError, "stages.revert.passes"):
            self.run_revert()

    def test_a_missing_count_file_says_what_to_run(self):
        job = job_with(self.cfg, **{"03_tagged.txt": ENGLISH, "05_winner.txt": WINNER})
        with self.assertRaises(FileNotFoundError):
            revert.run(self.cfg, job)

    def test_story_text_that_looks_like_a_placeholder_is_not_expanded(self):
        template = (Path(self.cfg["paths"]["prompts"]) / "revert.txt").read_text(encoding="utf-8")
        prompt = revert.build_prompt(template, "Type {{TARGET}} here.", "譯文{{NEW_CHARS}}", [], 5)
        self.assertIn("Type {{TARGET}} here.", prompt)
        self.assertIn("譯文{{NEW_CHARS}}", prompt)

    def test_new_characters_are_listed_not_preferred_first(self):
        text = revert.format_new_chars([{"char": "甲", "count": 5, "in_pool": True, "context": "甲乙"},
                                        {"char": "乙", "count": 1, "in_pool": False, "context": "乙丙"}])
        first, second = text.split("\n")
        self.assertTrue(first.startswith("乙") and "not on the preferred list" in first and "once" in first)
        self.assertTrue(second.startswith("甲") and "5 times" in second)


class Comprehension(Base):
    def revert_done(self, *replies):
        self.run_revert(*replies)

    def pick(self, n):
        texts = {1: GOOD_1, 2: GOOD_2, 3: GOOD_3}

        def handler(body, i):
            prompt = body["messages"][0]["content"]
            shown = tags.strip_all_tags(texts[n])
            letter = re.search(r"CANDIDATE ([A-C]) [^\n]*\n" + re.escape(shown), prompt).group(1)
            return f"REASON: reads best.\nWINNER: {letter}"
        return handler

    def test_the_judge_names_a_valid_reversion_and_it_becomes_08_winner(self):
        self.revert_done(GOOD_1, GOOD_2, GOOD_3)
        with FakeLLM(handler=self.pick(2)) as fake:
            note = judge.run_comprehension(self.cfg, self.job)
        self.assertEqual(self.job.read("08_winner.txt"), GOOD_2 + "\n")
        self.assertEqual(self.job.read_json("08_judge.json")["winner"], 2)
        self.assertIn("winner: reversion 2", note)
        self.assertIn("reads best", note)
        self.assertTrue(fake.calls[0]["url"].startswith("http://fake-cloud"))

    def test_each_reversion_can_win(self):
        self.revert_done(GOOD_1, GOOD_2, GOOD_3)
        for n in (1, 2, 3):
            with self.subTest(n=n):
                with FakeLLM(handler=self.pick(n)):
                    judge.run_comprehension(self.cfg, self.job)
                self.assertEqual(self.job.read_json("08_judge.json")["winner"], n)

    def test_the_judge_sees_the_english_the_rubric_the_counts_and_no_tags(self):
        self.revert_done(GOOD_1, GOOD_2, GOOD_3)
        with FakeLLM("WINNER: A") as fake:
            judge.run_comprehension(self.cfg, self.job)
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertIn("Dana walked to the market and bought apples near the river.", prompt)
        self.assertNotIn("<en>", prompt)
        self.assertNotIn("<dnt>", prompt)
        self.assertIn("Meaning:", prompt)
        self.assertIn("English words put back: 2", prompt)
        self.assertNotIn("{{", prompt)

    def test_rejected_reversions_are_not_candidates(self):
        self.revert_done(WINNER, GOOD_2, GOOD_3)
        with FakeLLM("WINNER: A") as fake:
            judge.run_comprehension(self.cfg, self.job)
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(len(re.findall(r"CANDIDATE [ABC] ", prompt)), 2)
        self.assertIn(self.job.read_json("08_judge.json")["winner"], (2, 3))

    def test_a_single_valid_reversion_wins_without_a_call(self):
        self.revert_done(WINNER, GOOD_2, WINNER)
        with FakeLLM() as fake:
            note = judge.run_comprehension(self.cfg, self.job)
        self.assertEqual(fake.calls, [])
        self.assertEqual(self.job.read("08_winner.txt"), GOOD_2 + "\n")
        self.assertIn("no judge call", note)

    def test_with_no_valid_reversion_the_translation_is_kept(self):
        self.revert_done(WINNER, WINNER, WINNER)
        with FakeLLM() as fake:
            note = judge.run_comprehension(self.cfg, self.job)
        self.assertEqual(fake.calls, [])
        self.assertEqual(self.job.read("08_winner.txt"), WINNER + "\n")
        self.assertIn("translation from 05_winner.txt is kept", note)
        self.assertEqual(self.job.read_json("08_judge.json")["method"], "kept_translation")

    def test_unreadable_answers_fall_back_to_the_middle_valid_reversion(self):
        self.revert_done(GOOD_1, GOOD_2, GOOD_3)
        with FakeLLM("no idea", "still none") as fake:
            note = judge.run_comprehension(self.cfg, self.job)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(self.job.read_json("08_judge.json")["winner"], 2)
        self.assertIn("WARNING", note)

    def test_a_missing_rubric_is_a_plain_error(self):
        self.revert_done(GOOD_1, GOOD_2, GOOD_3)
        (Path(self.cfg["paths"]["prompts"]) / "judge_comprehension_rubric.txt").unlink()
        with self.assertRaisesRegex(ConfigError, "judge_comprehension_rubric.txt"):
            with FakeLLM("WINNER: A"):
                judge.run_comprehension(self.cfg, self.job)

    def test_the_translation_judge_still_works_in_the_same_module(self):
        self.assertTrue(callable(judge.run_translation))


class ThroughThePipeline(Base):
    def go(self, name, *replies):
        self.job.mark_stage("count")
        self.job.mark_stage("judge_translation")
        with FakeLLM(*replies):
            return pipeline.run_stage(self.cfg, self.job, name, out=lambda s: None)

    def test_revert_runs_when_needed_and_is_marked_done(self):
        self.assertEqual(self.go("revert", GOOD_1, GOOD_2, GOOD_3), "done")
        self.assertTrue(self.job.stage_done("revert") and self.job.exists("07_meta.json"))

    def test_revert_and_the_judge_are_skipped_when_the_count_is_within_tolerance(self):
        self.job.write("05_winner.txt", "".join(KNOWN[:6]) + "。\n")
        count.run(self.cfg, self.job)
        self.assertEqual(self.go("revert"), "not_needed")
        self.assertEqual(self.go("judge_comprehension"), "not_needed")

    def test_a_conditional_stage_asks_for_the_count_when_it_has_not_run(self):
        job = job_with(self.cfg, **{"03_tagged.txt": ENGLISH, "05_winner.txt": WINNER})
        with self.assertRaisesRegex(pipeline.StageError, r"revert needs 06_count.json.*run.py count --job"):
            pipeline.run_stage(self.cfg, job, "revert", out=lambda s: None)

    def test_rerunning_revert_removes_the_old_reversion_and_final_files(self):
        self.go("revert", GOOD_1, GOOD_2, GOOD_3)
        with FakeLLM(handler=lambda b, i: "WINNER: A"):
            pipeline.run_stage(self.cfg, self.job, "judge_comprehension", out=lambda s: None)
        self.assertTrue(self.job.exists("08_winner.txt"))
        with FakeLLM(GOOD_1, GOOD_2, GOOD_3):
            pipeline.run_stage(self.cfg, self.job, "revert", force=True, out=lambda s: None)
        self.assertFalse(self.job.exists("08_winner.txt"))
        self.assertFalse(self.job.stage_done("judge_comprehension"))


if __name__ == "__main__":
    unittest.main()
