"""Step 3 tests: stages/tag.py (<dnt> tagging), and simplify then tag run through pipeline.py, with a
scripted fake model (no LM Studio needed).

Run from the project folder:
    python3 tests/test_tag.py -v
    python3 -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core import textcheck                         # noqa: E402
from stages import tag                             # noqa: E402
from tests.fakes import FakeLLM                    # noqa: E402
from tests.helpers import DNT_PROMPT, SIMPLIFY_PROMPT, job_with, stage_cfg   # noqa: E402

FIX = BASE / "tests" / "fixtures"
STORY = ("Dana Whitfield, the mayor of Springfield, said the plan is good. She spoke to the BBC on Tuesday.\n\n"
         "The lanes will cost about eighteen million dollars.")
TAGGED = ("<dnt>Dana Whitfield</dnt>, the mayor of <dnt>Springfield</dnt>, said the plan is good. "
          "She spoke to the <dnt>BBC</dnt> on Tuesday.\n\n"
          "The lanes will cost about eighteen million dollars.")


class CheckTagging(unittest.TestCase):
    def test_a_correct_reply_is_accepted_unchanged(self):
        tagged, problem = tag.check_tagging(STORY, TAGGED)
        self.assertEqual((tagged, problem), (TAGGED, ""))

    def test_a_code_fence_and_an_introduction_line_are_removed(self):
        tagged, problem = tag.check_tagging(STORY, "```\nHere is the tagged text:\n" + TAGGED + "\n```")
        self.assertEqual((tagged, problem), (TAGGED, ""))

    def test_a_reply_with_no_tags_at_all_is_valid(self):
        tagged, problem = tag.check_tagging(STORY, STORY)
        self.assertEqual((tagged, problem), (STORY, ""))

    def test_different_spacing_and_line_breaks_are_accepted(self):
        flattened = TAGGED.replace("\n\n", " ")
        tagged, problem = tag.check_tagging(STORY, flattened)
        self.assertEqual(problem, "")
        self.assertEqual(tagged, flattened)

    def test_an_opening_tag_with_no_closing_tag_is_unbalanced(self):
        bad = TAGGED.replace("</dnt>", "", 1)
        self.assertEqual(tag.check_tagging(STORY, bad), (None, "the tags were unbalanced"))

    def test_a_closing_tag_with_no_opening_tag_is_unbalanced(self):
        bad = TAGGED.replace("<dnt>", "", 1)
        self.assertEqual(tag.check_tagging(STORY, bad), (None, "the tags were unbalanced"))

    def test_nested_tags_are_unbalanced(self):
        bad = TAGGED.replace("<dnt>BBC</dnt>", "<dnt>the <dnt>BBC</dnt></dnt>")
        self.assertEqual(tag.check_tagging(STORY, bad)[1], "the tags were unbalanced")

    def test_changed_wording_is_rejected(self):
        bad = TAGGED.replace("said the plan is good", "said the plan was great")
        self.assertEqual(tag.check_tagging(STORY, bad), (None, "the model changed the wording"))

    def test_a_dropped_sentence_is_rejected(self):
        bad = TAGGED.split("\n\n")[0]
        self.assertEqual(tag.check_tagging(STORY, bad)[1], "the model changed the wording")

    def test_extra_text_added_by_the_model_is_rejected(self):
        bad = TAGGED + "\n\nNOTE: I tagged three names."
        self.assertEqual(tag.check_tagging(STORY, bad)[1], "the model changed the wording")

    def test_empty_tag_pairs_are_dropped_not_rejected(self):
        tagged, problem = tag.check_tagging(STORY, TAGGED.replace("on Tuesday", "on <dnt></dnt>Tuesday"))
        self.assertEqual(problem, "")
        self.assertNotIn("<dnt></dnt>", tagged)

    def test_uppercase_tags_are_not_tags_and_count_as_changed_wording(self):
        bad = TAGGED.replace("<dnt>", "<DNT>").replace("</dnt>", "</DNT>")
        self.assertEqual(tag.check_tagging(STORY, bad)[1], "the model changed the wording")

    def test_the_real_starlink_story(self):
        tagged_story = (FIX / "starlink_tagged_story.txt").read_text(encoding="utf-8")
        plain = textcheck.strip_dnt_tags(tagged_story)
        tagged, problem = tag.check_tagging(plain, tagged_story)
        self.assertEqual(problem, "")
        spans = textcheck.extract_dnt(tagged)
        for name in ("Serhiy Beskrestnov", "Elon Musk", "SpaceX", "Starlink", "Financial Times", "U.S."):
            self.assertIn(name, spans)


class TagStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        self.job = job_with(self.cfg, **{"02_simplified.txt": STORY + "\n"})

    def run_stage(self, reply):
        with FakeLLM(reply) as fake:
            note = tag.run(self.cfg, self.job)
        return note, fake

    def test_a_good_reply_is_saved_with_its_tags(self):
        note, _ = self.run_stage(TAGGED)
        self.assertEqual(self.job.read("03_tagged.txt"), TAGGED + "\n")
        self.assertEqual(note, "protected 3 spans: Dana Whitfield, Springfield, BBC")
        meta = self.job.status()["meta"]
        self.assertEqual((meta["tagging"], meta["protected_spans"]), ("ok", 3))
        self.assertFalse(self.job.exists("03_tag_reply_raw.txt"))

    def test_the_prompt_and_story_go_to_the_model_as_one_user_message(self):
        _, fake = self.run_stage(TAGGED)
        messages = fake.calls[0]["body"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], DNT_PROMPT + "\n\n" + STORY)

    def test_uses_the_profile_the_stage_names_not_the_one_simplify_uses(self):
        self.cfg["stages"]["tag"]["model"] = "local2"
        _, fake = self.run_stage(TAGGED)
        self.assertEqual(fake.calls[0]["body"]["model"], "fake-model-2")

    def test_a_long_story_raises_the_reply_token_limit(self):
        job = job_with(self.cfg, **{"02_simplified.txt": "word " * 5000})
        with FakeLLM("word " * 5000) as fake:
            tag.run(self.cfg, job)
        self.assertGreater(fake.calls[0]["body"]["max_tokens"], 4096)

    def test_unbalanced_tags_are_thrown_away_and_the_plain_story_is_kept(self):
        note, _ = self.run_stage(TAGGED.replace("</dnt>", "", 1))
        self.assertEqual(self.job.read("03_tagged.txt"), STORY + "\n")
        self.assertIn("WARNING: tagging discarded (the tags were unbalanced)", note)
        meta = self.job.status()["meta"]
        self.assertEqual((meta["tagging"], meta["tagging_problem"]), ("discarded", "the tags were unbalanced"))
        self.assertIn("<dnt>", self.job.read("03_tag_reply_raw.txt"))
        self.assertIn("tag: discarded", self.job.read("log.txt"))

    def test_changed_wording_is_thrown_away_too(self):
        note, _ = self.run_stage(TAGGED.replace("good", "excellent"))
        self.assertEqual(self.job.read("03_tagged.txt"), STORY + "\n")
        self.assertIn("the model changed the wording", note)

    def test_a_reply_with_no_tags_means_nothing_needed_protecting(self):
        note, _ = self.run_stage(STORY)
        self.assertEqual(note, "no text needed protecting")
        self.assertEqual(self.job.status()["meta"]["tagging"], "ok")

    def test_a_good_rerun_clears_the_raw_reply_and_the_old_problem(self):
        self.run_stage(TAGGED.replace("</dnt>", "", 1))
        self.assertTrue(self.job.exists("03_tag_reply_raw.txt"))
        self.run_stage(TAGGED)
        self.assertFalse(self.job.exists("03_tag_reply_raw.txt"))
        meta = self.job.status()["meta"]
        self.assertEqual((meta["tagging"], meta["tagging_problem"]), ("ok", ""))

    def test_more_than_eight_spans_are_summarised(self):
        story = " ".join(f"Name{i} spoke." for i in range(12))
        job = job_with(self.cfg, **{"02_simplified.txt": story})
        reply = " ".join(f"<dnt>Name{i}</dnt> spoke." for i in range(12))
        with FakeLLM(reply):
            note = tag.run(self.cfg, job)
        self.assertTrue(note.startswith("protected 12 spans: Name0, Name1"))
        self.assertTrue(note.endswith("..."))

    def test_the_story_text_in_03_never_differs_from_02_once_tags_are_removed(self):
        for reply in (TAGGED, TAGGED.replace("good", "great"), TAGGED.replace("</dnt>", "", 1), STORY):
            with self.subTest(reply=reply[:30]):
                self.run_stage(reply)
                kept = textcheck.strip_dnt_tags(self.job.read("03_tagged.txt"))
                self.assertEqual(" ".join(kept.split()), " ".join(STORY.split()))


class SimplifyThenTagThroughThePipeline(unittest.TestCase):
    """simplify and tag run by pipeline.run_stage, the way run.py runs them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        self.job = job_with(self.cfg, **{"01_clean.txt": STORY + "\n"})
        self.out = []

    def go(self, name, reply):
        with FakeLLM(reply) as fake:
            result = pipeline.run_stage(self.cfg, self.job, name, out=self.out.append)
        return result, fake

    def test_both_stages_finish_and_leave_the_files_the_next_stage_needs(self):
        self.assertEqual(self.go("simplify", STORY)[0], "done")
        self.assertEqual(self.go("tag", TAGGED)[0], "done")
        self.assertTrue(self.job.stage_done("simplify") and self.job.stage_done("tag"))
        self.assertTrue(self.job.exists("02_simplified.txt") and self.job.exists("03_tagged.txt"))
        self.assertTrue(any("tag: done" in line and "protected 3 spans" in line for line in self.out))

    def test_tag_cannot_run_before_simplify_and_says_what_to_run(self):
        with self.assertRaisesRegex(pipeline.StageError, r"tag needs 02_simplified.txt.*run.py simplify --job"):
            pipeline.run_stage(self.cfg, self.job, "tag", out=self.out.append)

    def test_a_finished_stage_is_skipped_on_a_rerun_without_calling_the_model(self):
        self.go("simplify", STORY)
        with FakeLLM() as fake:                                       # no replies scripted: a call would fail
            result = pipeline.run_stage(self.cfg, self.job, "simplify", out=self.out.append)
        self.assertEqual((result, fake.calls), ("skipped", []))

    def test_rerunning_simplify_forgets_the_tag_result(self):
        self.go("simplify", STORY)
        self.go("tag", TAGGED)
        with FakeLLM(STORY):
            pipeline.run_stage(self.cfg, self.job, "simplify", force=True, out=self.out.append)
        self.assertFalse(self.job.stage_done("tag"))

    def test_a_failed_model_call_marks_the_stage_failed_with_a_readable_reason(self):
        with FakeLLM(RuntimeError("The model server returned an error: boom")):
            with self.assertRaisesRegex(pipeline.StageError, "simplify failed: The model server returned an error"):
                pipeline.run_stage(self.cfg, self.job, "simplify", out=self.out.append)
        self.assertEqual(self.job.status()["stages"]["simplify"]["status"], "failed")

    def test_the_simplify_prompt_is_not_the_tag_prompt(self):
        _, fake_s = self.go("simplify", STORY)
        _, fake_t = self.go("tag", TAGGED)
        self.assertEqual(fake_s.calls[0]["body"]["messages"][0]["content"], SIMPLIFY_PROMPT)
        self.assertTrue(fake_t.calls[0]["body"]["messages"][0]["content"].startswith(DNT_PROMPT))


if __name__ == "__main__":
    unittest.main()
