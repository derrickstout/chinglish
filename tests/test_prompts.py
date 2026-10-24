"""Step 3 tests: core/prompts.py (prompt files, reply token budgets, and tidying a model's reply).

Run from the project folder:
    python3 tests/test_prompts.py -v
    python3 -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import prompts                           # noqa: E402
from core.config import ConfigError                # noqa: E402


class PromptFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Pretend the project folder is the temporary one, so the real project's folders are never read.
        patcher = mock.patch.object(prompts, "BASE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def folder(self, name, **files):
        d = self.root / name
        d.mkdir()
        for fname, text in files.items():
            (d / fname.replace("__", ".")).write_text(text, encoding="utf-8")
        return d

    def test_loads_a_prompt_from_the_configured_folder_and_strips_blank_space(self):
        d = self.folder("my_prompts", simplify__txt="\n  Make it simple.  \n\n")
        cfg = {"paths": {"prompts": str(d)}}
        self.assertEqual(prompts.load_prompt(cfg, "simplify.txt"), "Make it simple.")

    def test_a_relative_folder_is_under_the_project(self):
        self.folder("prompts", dnt__txt="Tag names.")
        self.assertEqual(prompts.load_prompt({"paths": {"prompts": "prompts"}}, "dnt.txt"), "Tag names.")

    def test_falls_back_to_a_capitalised_folder_name(self):
        self.folder("Prompts", dnt__txt="Tag names.")
        self.assertEqual(prompts.load_prompt({"paths": {"prompts": "prompts"}}, "dnt.txt"), "Tag names.")

    def test_no_folder_is_a_plain_message(self):
        with self.assertRaisesRegex(ConfigError, "prompts folder was not found"):
            prompts.load_prompt({"paths": {"prompts": "nowhere"}}, "dnt.txt")

    def test_a_missing_file_names_the_file_and_the_folder(self):
        d = self.folder("prompts")
        with self.assertRaisesRegex(ConfigError, r"dnt\.txt is not in"):
            prompts.load_prompt({"paths": {"prompts": str(d)}}, "dnt.txt")

    def test_an_empty_file_is_an_error(self):
        d = self.folder("prompts", dnt__txt="  \n")
        with self.assertRaisesRegex(ConfigError, "is empty"):
            prompts.load_prompt({"paths": {"prompts": str(d)}}, "dnt.txt")

    def test_a_file_that_is_not_utf8_is_an_error(self):
        d = self.folder("prompts")
        (d / "dnt.txt").write_bytes(b"\xff\xfe\x00bad")
        with self.assertRaisesRegex(ConfigError, "Could not read"):
            prompts.load_prompt({"paths": {"prompts": str(d)}}, "dnt.txt")


def budget_cfg(max_tokens, **stage):
    return {"models": {"m": {"url": "http://x/v1", "model": "x", "max_tokens": max_tokens}},
            "stages": {"simplify": dict({"model": "m"}, **stage)}}


class TokenBudget(unittest.TestCase):
    def test_a_short_text_keeps_the_models_limit(self):
        self.assertEqual(prompts.token_budget(budget_cfg(4096), "simplify", "a short story"), 4096)

    def test_a_long_text_raises_the_limit(self):
        self.assertGreater(prompts.token_budget(budget_cfg(4096), "simplify", "word " * 4000), 4096)

    def test_the_limit_never_drops_below_the_models_limit(self):
        self.assertEqual(prompts.token_budget(budget_cfg(9000), "simplify", "x" * 100), 9000)

    def test_a_longer_expected_reply_gets_more_room(self):
        cfg, text = budget_cfg(100), "x" * 20000
        self.assertGreater(prompts.token_budget(cfg, "simplify", text, factor=1.4), prompts.token_budget(cfg, "simplify", text))

    def test_a_missing_max_tokens_uses_the_default(self):
        cfg = {"models": {"m": {"url": "http://x/v1", "model": "x"}}, "stages": {"simplify": {"model": "m"}}}
        self.assertGreaterEqual(prompts.token_budget(cfg, "simplify", "x"), 4096)

    def test_the_stage_override_is_used_exactly_even_for_a_long_text(self):
        cfg = budget_cfg(4096, max_tokens_override=555)
        self.assertEqual(prompts.token_budget(cfg, "simplify", "word " * 9000), 555)

    def test_the_stage_decides_which_profile_is_used(self):
        cfg = {"models": {"a": {"url": "u", "model": "x", "max_tokens": 5111}, "b": {"url": "u", "model": "x", "max_tokens": 6222}},
               "stages": {"simplify": {"model": "a"}, "tag": {"model": "b"}}}
        self.assertEqual(prompts.token_budget(cfg, "simplify", "x"), 5111)
        self.assertEqual(prompts.token_budget(cfg, "tag", "x"), 6222)


class StripWrapper(unittest.TestCase):
    def test_removes_a_code_fence_around_the_whole_reply(self):
        text, notes = prompts.strip_wrapper("```text\nHello there.\n```")
        self.assertEqual(text, "Hello there.")
        self.assertEqual(notes, ["removed a code fence"])

    def test_removes_an_introduction_line_ending_in_a_colon(self):
        text, notes = prompts.strip_wrapper("Here is the simplified text:\n\nThe city voted.")
        self.assertEqual(text, "The city voted.")
        self.assertEqual(notes, ["removed an introduction line"])

    def test_removes_both(self):
        text, notes = prompts.strip_wrapper("```\nSure, here you go:\nThe city voted.\n```")
        self.assertEqual(text, "The city voted.")
        self.assertEqual(len(notes), 2)

    def test_leaves_a_normal_reply_alone(self):
        text, notes = prompts.strip_wrapper("The city voted.\n\nIt was close.")
        self.assertEqual(text, "The city voted.\n\nIt was close.")
        self.assertEqual(notes, [])

    def test_does_not_remove_a_first_sentence_that_merely_starts_with_here(self):
        text, notes = prompts.strip_wrapper("Here is the thing about bikes. They are quiet.\nPeople like them.")
        self.assertTrue(text.startswith("Here is the thing"))
        self.assertEqual(notes, [])

    def test_an_introduction_alone_is_not_removed_if_nothing_follows(self):
        text, notes = prompts.strip_wrapper("Here is the text:")
        self.assertEqual(text, "Here is the text:")


if __name__ == "__main__":
    unittest.main()
