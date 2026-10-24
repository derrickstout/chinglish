"""Step 5 tests: stages/judge.py (judge_translation), with a scripted fake cloud model.

Run from the project folder:
    python3 tests/test_judge.py -v
    python3 -m unittest discover tests
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core.config import ConfigError                # noqa: E402
from stages import judge                           # noqa: E402
from tests.fakes import FakeLLM, http_error        # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

T = [("<dnt>Dana</dnt>去了市場。", 3), ("<dnt>Dana</dnt>去市場。", 5), ("<dnt>Dana</dnt>到市場去了。", 9)]
ENGLISH = "<dnt>Dana</dnt> went to the market."


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        prompts = Path(self.cfg["paths"]["prompts"])
        shutil_src = BASE / "prompts"
        for name in ("judge_translation.txt", "judge_rubric.txt"):
            (prompts / name).write_text((shutil_src / name).read_text(encoding="utf-8"), encoding="utf-8")
        env = mock.patch.dict(os.environ, {"TEST_CLOUD_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)
        self.job = self.make_job(T)

    def make_job(self, items):
        files = {"03_tagged.txt": ENGLISH + "\n"}
        cands = []
        for n, (text, new) in enumerate(items, 1):
            files[f"04_translation_{n}.txt"] = text + "\n"
            cands.append({"n": n, "temperature": [0.3, 0.6, 0.9][n - 1], "file": f"04_translation_{n}.txt",
                          "error": None, "new_chars": new})
        job = job_with(self.cfg, **files)
        job.write_json("04_meta.json", {"candidates": cands})
        return job

    def run_stage(self, *replies, job=None):
        with FakeLLM(*replies) as fake:
            note = judge.run_translation(self.cfg, job or self.job)
        return note, fake

    def label_for(self, fake, n, call=0):
        """The letter the judge was shown for translation n (found by its text in the prompt)."""
        prompt = fake.calls[call]["body"]["messages"][0]["content"]
        text = re.sub(r"</?dnt>", "", T[n - 1][0])
        return re.search(r"CANDIDATE ([A-C]) [^\n]*\n" + re.escape(text), prompt).group(1)

    def pick(self, n):
        """A handler that answers with the label shown for translation n."""
        def handler(body, i):
            prompt = body["messages"][0]["content"]
            text = re.sub(r"</?dnt>", "", T[n - 1][0])
            letter = re.search(r"CANDIDATE ([A-C]) [^\n]*\n" + re.escape(text), prompt).group(1)
            return f"REASON: it is accurate.\nWINNER: {letter}"
        return handler


class JudgeStage(Base):
    def test_the_winner_the_judge_names_is_saved_with_its_tags(self):
        with FakeLLM(handler=self.pick(3)) as fake:
            note = judge.run_translation(self.cfg, self.job)
        self.assertEqual(self.job.read("05_winner.txt"), T[2][0] + "\n")
        self.assertIn("winner: translation 3", note)
        self.assertIn("it is accurate", note)
        info = self.job.read_json("05_judge.json")
        self.assertEqual((info["winner"], info["method"]), (3, "judge"))

    def test_each_translation_can_win_whatever_its_label(self):
        for n in (1, 2, 3):
            with self.subTest(n=n):
                with FakeLLM(handler=self.pick(n)):
                    judge.run_translation(self.cfg, self.job)
                self.assertEqual(self.job.read_json("05_judge.json")["winner"], n)

    def test_the_call_goes_to_the_cloud_model_at_temperature_zero(self):
        _, fake = self.run_stage("WINNER: A")
        call = fake.calls[0]
        self.assertTrue(call["url"].startswith("http://fake-cloud/v1"))
        self.assertEqual(call["body"]["model"], "judge-model")
        self.assertEqual(call["body"]["temperature"], 0.0)

    def test_the_judge_sees_the_english_the_rubric_and_untagged_candidates_with_counts(self):
        _, fake = self.run_stage("WINNER: B")
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertIn("Dana went to the market.", prompt)
        self.assertNotIn("<dnt>", prompt)
        self.assertIn("Accuracy", prompt)
        for count in (3, 5, 9):
            self.assertIn(f"known list: {count})", prompt)
        self.assertNotIn("{{", prompt)

    def test_the_labels_hide_the_temperatures(self):
        _, fake = self.run_stage("WINNER: A")
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertNotIn("0.3", prompt)
        self.assertNotIn("temperature", prompt.lower())

    def test_the_order_is_shuffled_but_the_same_for_the_same_job(self):
        orders = set()
        for k in range(12):
            orders.add(tuple(judge.shuffled(range(3), f"seed{k}")))
        self.assertGreater(len(orders), 1)
        self.assertEqual(judge.shuffled(range(3), "x"), judge.shuffled(range(3), "x"))

    def test_the_reply_format_is_forgiving(self):
        for reply, want in (("WINNER: B", 1), ("**WINNER: C**", 2), ("reason\nWinner: a.", 0),
                            ("WINNER: [B]", 1), ("WINNER: A\nthen again\nWINNER: C", 2)):
            with self.subTest(reply=reply):
                self.assertEqual(judge.parse_winner(reply, 3), want)

    def test_unreadable_replies_give_none(self):
        for reply in ("I like the first one.", "WINNER: D", "WINNER:", ""):
            with self.subTest(reply=reply):
                self.assertIsNone(judge.parse_winner(reply, 3))

    def test_an_unreadable_reply_is_retried_once(self):
        note, fake = self.run_stage("Hmm, hard to say.", "WINNER: A")
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn("WARNING", note)

    def test_two_unreadable_replies_fall_back_to_the_middle_temperature(self):
        note, fake = self.run_stage("no idea", "still no idea")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(self.job.read("05_winner.txt"), T[1][0] + "\n")
        self.assertIn("WARNING", note)
        self.assertEqual(self.job.read_json("05_judge.json")["method"], "fallback")

    def test_a_failed_call_is_retried_and_then_falls_back(self):
        note, fake = self.run_stage(http_error(500, "x"), http_error(500, "y"))
        self.assertEqual(self.job.read_json("05_judge.json")["winner"], 2)
        self.assertIn("WARNING", note)

    def test_the_number_of_retries_comes_from_settings(self):
        for retries, calls in ((0, 1), (1, 2), (3, 4)):
            with self.subTest(retries=retries):
                self.cfg["stages"]["judge_translation"]["retries"] = retries
                with FakeLLM(*["no idea"] * calls) as fake:
                    judge.run_translation(self.cfg, self.job)
                self.assertEqual(len(fake.calls), calls)

    def test_the_judge_uses_the_profile_the_stage_names_and_its_temperature(self):
        self.cfg["stages"]["judge_translation"].update(model="local1", temperature=0.2)
        with FakeLLM("WINNER: A") as fake:
            judge.run_translation(self.cfg, self.job)
        self.assertTrue(fake.calls[0]["url"].startswith("http://fake:1234/v1"))
        self.assertEqual((fake.calls[0]["body"]["model"], fake.calls[0]["body"]["temperature"]), ("fake-model", 0.2))
        self.assertNotIn("Authorization", fake.calls[0]["headers"])

    def test_the_two_judges_can_use_different_profiles(self):
        self.cfg["stages"]["judge_comprehension"]["model"] = "local2"
        self.assertEqual(self.cfg["stages"]["judge_translation"]["model"], "cloud1")
        from core import schema
        self.assertEqual(schema.stage_opts(self.cfg, "judge_comprehension")["model"], "local2")

    def test_max_tokens_override_is_sent_when_set_and_otherwise_the_profile_limit_applies(self):
        with FakeLLM("WINNER: A") as fake:
            judge.run_translation(self.cfg, self.job)
        self.assertEqual(fake.calls[0]["body"]["max_tokens"], 2048)
        self.cfg["stages"]["judge_translation"]["max_tokens_override"] = 321
        with FakeLLM("WINNER: A") as fake:
            judge.run_translation(self.cfg, self.job)
        self.assertEqual(fake.calls[0]["body"]["max_tokens"], 321)

    def test_a_failed_call_then_a_good_one_works(self):
        note, _ = self.run_stage(http_error(503, "busy"), "WINNER: A")
        self.assertNotIn("WARNING", note)

    def test_a_missing_cloud_key_stops_at_once_with_the_fix(self):
        with mock.patch.dict(os.environ, {"TEST_CLOUD_KEY": ""}):
            with FakeLLM() as fake:
                with self.assertRaisesRegex(ConfigError, "export TEST_CLOUD_KEY"):
                    judge.run_translation(self.cfg, self.job)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.job.exists("05_winner.txt"))

    def test_with_one_translation_no_call_is_made(self):
        job = self.make_job(T[:1])
        note, fake = self.run_stage(job=job)
        self.assertEqual(fake.calls, [])
        self.assertEqual(job.read("05_winner.txt"), T[0][0] + "\n")
        self.assertIn("no judge call", note)

    def test_failed_translations_are_not_candidates(self):
        meta = self.job.read_json("04_meta.json")
        meta["candidates"][1].update(file=None, error="boom")
        self.job.write_json("04_meta.json", meta)
        with FakeLLM("WINNER: B") as fake:
            judge.run_translation(self.cfg, self.job)
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertNotIn(re.sub(r"</?dnt>", "", T[1][0]), prompt)
        self.assertIn(self.job.read_json("05_judge.json")["winner"], (1, 3))

    def test_identical_translations_still_get_their_own_labels(self):
        job = self.make_job([T[0], T[0], T[2]])
        _, fake = self.run_stage("WINNER: A", job=job)
        prompt = fake.calls[0]["body"]["messages"][0]["content"]
        self.assertEqual(len(re.findall(r"CANDIDATE [ABC] ", prompt)), 3)

    def test_no_translations_is_an_error_that_says_what_to_run(self):
        job = job_with(self.cfg, **{"03_tagged.txt": "x"})
        job.write_json("04_meta.json", {"candidates": [{"n": 1, "file": None, "error": "boom"}]})
        with self.assertRaisesRegex(RuntimeError, "run.py translate --job"):
            judge.run_translation(self.cfg, job)

    def test_a_missing_rubric_file_is_a_plain_error(self):
        (Path(self.cfg["paths"]["prompts"]) / "judge_rubric.txt").unlink()
        with self.assertRaisesRegex(ConfigError, "judge_rubric.txt"):
            self.run_stage("WINNER: A")

    def test_a_template_that_lost_a_placeholder_is_an_error(self):
        (Path(self.cfg["paths"]["prompts"]) / "judge_translation.txt").write_text("Pick one.", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "CANDIDATES"):
            self.run_stage("WINNER: A")

    def test_the_shipped_files_have_what_the_stage_needs(self):
        text = (BASE / "prompts" / "judge_translation.txt").read_text(encoding="utf-8")
        for p in ("{{ENGLISH}}", "{{CANDIDATES}}", "{{RUBRIC}}", "WINNER:"):
            self.assertIn(p, text)


class ThroughThePipeline(Base):
    def test_judge_translation_runs_after_translate_and_is_marked_done(self):
        self.job.mark_stage("translate")
        out = []
        with FakeLLM("WINNER: A"):
            result = pipeline.run_stage(self.cfg, self.job, "judge_translation", out=out.append)
        self.assertEqual(result, "done")
        self.assertTrue(self.job.stage_done("judge_translation"))
        self.assertTrue(any("judge_translation: done" in line for line in out))

    def test_the_judgement_is_forgotten_when_an_earlier_stage_is_rerun(self):
        self.job.mark_stage("translate")
        with FakeLLM("WINNER: A"):
            pipeline.run_stage(self.cfg, self.job, "judge_translation", out=lambda s: None)
        pipeline._invalidate_after(self.job, "translate")
        self.assertFalse(self.job.stage_done("judge_translation"))


if __name__ == "__main__":
    unittest.main()
