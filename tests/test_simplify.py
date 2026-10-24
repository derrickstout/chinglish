"""Step 3 tests: stages/simplify.py, with a scripted fake model (no LM Studio needed).

Run from the project folder:
    python3 tests/test_simplify.py -v
    python3 -m unittest discover tests
"""
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core.config import ConfigError                # noqa: E402
from stages import simplify                        # noqa: E402
from tests.fakes import FakeLLM, http_error        # noqa: E402
from tests.helpers import SIMPLIFY_PROMPT, job_with, stage_cfg   # noqa: E402

ARTICLE = ("The city council voted on Tuesday to expand the downtown bike lane network after months of "
           "public debate. The plan adds twelve miles of protected lanes and is expected to cost eighteen "
           "million dollars over three years. Supporters say the lanes will make streets safer.")
SIMPLE = ("The city council voted on Tuesday to add more bike lanes downtown. The plan adds twelve miles of "
          "safe lanes and will cost eighteen million dollars over three years. Supporters say this will "
          "make streets safer.")


class SimplifyStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        self.job = job_with(self.cfg, **{"01_clean.txt": ARTICLE + "\n"})

    def run_stage(self, *replies):
        with FakeLLM(*replies) as fake:
            note = simplify.run(self.cfg, self.job)
        return note, fake

    def test_writes_the_simplified_text_and_reports_word_counts(self):
        note, _ = self.run_stage(SIMPLE)
        self.assertEqual(self.job.read("02_simplified.txt"), SIMPLE + "\n")
        self.assertRegex(note, r"\d+ -> \d+ words")
        self.assertNotIn("WARNING", note)

    def test_sends_the_prompt_as_the_system_message_and_the_article_as_the_user_message(self):
        _, fake = self.run_stage(SIMPLE)
        messages = fake.calls[0]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SIMPLIFY_PROMPT)
        self.assertEqual(messages[1]["content"], ARTICLE)

    def test_goes_to_the_local_model(self):
        _, fake = self.run_stage(SIMPLE)
        self.assertTrue(fake.calls[0]["url"].startswith("http://fake:1234/v1"))
        self.assertEqual(fake.calls[0]["body"]["model"], "fake-model")

    def test_uses_the_profile_and_temperature_the_stage_names(self):
        self.cfg["stages"]["simplify"].update(model="local2", temperature=0.1)
        _, fake = self.run_stage(SIMPLE)
        self.assertEqual((fake.calls[0]["body"]["model"], fake.calls[0]["body"]["temperature"]), ("fake-model-2", 0.1))

    def test_without_a_stage_temperature_the_profiles_is_used(self):
        _, fake = self.run_stage(SIMPLE)
        self.assertEqual(fake.calls[0]["body"]["temperature"], 0.3)

    def test_asks_the_model_manager_to_switch_models_once_before_the_call(self):
        with mock.patch("core.llm.ensure_model_switched") as switched:
            with FakeLLM(SIMPLE):
                simplify.run(self.cfg, self.job)
        switched.assert_called_once_with(self.cfg, "local1")

    def test_a_long_article_raises_the_reply_token_limit(self):
        job = job_with(self.cfg, **{"01_clean.txt": ("word " * 6000)})
        with FakeLLM(SIMPLE) as fake:
            simplify.run(self.cfg, job)
        self.assertGreater(fake.calls[0]["body"]["max_tokens"], 4096)

    def test_removes_a_code_fence_an_introduction_and_markdown(self):
        reply = "```\nHere is the simplified text:\n\n**The city** voted.\n\n## Lanes\nIt will cost money.\n```"
        note, _ = self.run_stage(reply)
        self.assertEqual(self.job.read("02_simplified.txt"), "The city voted.\n\nLanes\nIt will cost money.\n")
        self.assertIn("removed a code fence", note)
        self.assertIn("removed an introduction line", note)

    def test_a_much_shorter_reply_is_a_warning_not_an_error(self):
        note, _ = self.run_stage("The city voted.")
        self.assertIn("WARNING", note)
        self.assertIn("only", note)
        self.assertTrue(self.job.exists("02_simplified.txt"))
        self.assertIn("simplify warning", self.job.read("log.txt"))

    def test_a_reply_that_stops_mid_sentence_warns_that_it_may_be_cut_off(self):
        note, _ = self.run_stage(SIMPLE.rsplit(" ", 3)[0])
        self.assertIn("does not end with a full stop", note)

    def test_chinese_in_the_reply_warns_that_the_model_translated(self):
        note, _ = self.run_stage(SIMPLE + " 市議會投票了。")
        self.assertIn("contains Chinese characters", note)

    def test_a_much_longer_reply_warns(self):
        note, _ = self.run_stage(SIMPLE + " " + SIMPLE + " " + SIMPLE)
        self.assertIn("% of the original length", note)

    def test_an_empty_reply_after_cleanup_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.run_stage("```\n\n```")
        self.assertFalse(self.job.exists("02_simplified.txt"))

    def test_a_server_error_is_passed_on_in_plain_words(self):
        with self.assertRaisesRegex(RuntimeError, "returned an error"):
            self.run_stage(http_error(500, "model crashed"))

    def test_an_unreachable_server_is_passed_on_in_plain_words(self):
        with self.assertRaisesRegex(RuntimeError, "Could not reach"):
            self.run_stage(urllib.error.URLError("connection refused"))

    def test_a_missing_prompt_file_is_a_plain_error_and_no_model_call_is_made(self):
        (Path(self.cfg["paths"]["prompts"]) / "simplify.txt").unlink()
        with FakeLLM() as fake:
            with self.assertRaisesRegex(ConfigError, r"simplify\.txt"):
                simplify.run(self.cfg, self.job)
        self.assertEqual(fake.calls, [])

    def test_a_missing_input_file_says_which_stage_writes_it(self):
        job = job_with(self.cfg)
        with self.assertRaisesRegex(FileNotFoundError, "01_clean.txt not found"):
            simplify.run(self.cfg, job)

    def test_surrounding_blank_space_in_the_article_is_not_sent(self):
        job = job_with(self.cfg, **{"01_clean.txt": "\n\n" + ARTICLE + "\n\n\n"})
        with FakeLLM(SIMPLE) as fake:
            simplify.run(self.cfg, job)
        self.assertEqual(fake.calls[0]["body"]["messages"][1]["content"], ARTICLE)


class TidyAndCheck(unittest.TestCase):
    def test_tidy_leaves_plain_text_and_single_blank_lines_alone(self):
        text, notes = simplify.tidy("One.\n\nTwo.")
        self.assertEqual((text, notes), ("One.\n\nTwo.", []))

    def test_tidy_collapses_long_runs_of_blank_lines(self):
        text, _ = simplify.tidy("One.\n\n\n\n\nTwo.")
        self.assertEqual(text, "One.\n\nTwo.")

    def test_a_short_text_never_triggers_the_length_warnings(self):
        self.assertEqual(simplify.check_reply("Short text here.", "Shorter."), [])

    def test_a_clean_reply_has_no_warnings(self):
        self.assertEqual(simplify.check_reply(ARTICLE, SIMPLE), [])


if __name__ == "__main__":
    unittest.main()
