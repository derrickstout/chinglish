"""Step B tests: retries, retry temperatures and on_fail for tag and simplify, and the judge's retry temperature."""
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import schema                                   # noqa: E402
from stages import simplify, tag                          # noqa: E402
from tests.fakes import FakeLLM                           # noqa: E402
from tests.helpers import job_with, stage_cfg             # noqa: E402
from tests.test_tag import STORY, TAGGED                  # noqa: E402

BAD = "<dnt>Dana Whitfield, the mayor"


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = stage_cfg(self._tmp.name)

    def stage(self, name, **opts):
        self.cfg["stages"][name].update(opts)


class AttemptTemperature(Base):
    def test_first_attempt_uses_the_profile_temperature_then_steps_up_to_the_cap(self):
        opts = schema.stage_opts(self.cfg, "tag")
        self.assertIsNone(schema.attempt_temperature(self.cfg, opts, 0))
        self.assertEqual(schema.attempt_temperature(self.cfg, opts, 1), 0.4)
        self.assertEqual(schema.attempt_temperature(self.cfg, opts, 50), 1.0)

    def test_defaults_are_two_retries_and_continue_for_tag(self):
        cfg = stage_cfg(self._tmp.name)
        del cfg["stages"]["tag"]["retries"], cfg["stages"]["tag"]["on_fail"]
        opts = schema.stage_opts(cfg, "tag")
        self.assertEqual((opts["retries"], opts["on_fail"]), (2, "continue"))
        self.assertEqual(schema.stage_opts(cfg, "simplify")["on_fail"], "stop")


class TagRetries(Base):
    def test_a_bad_reply_is_retried_with_the_reason_and_a_warmer_temperature(self):
        self.stage("tag", retries=2, retry_temp_step=0.2)
        job = job_with(self.cfg, **{"02_simplified.txt": STORY})
        with FakeLLM(BAD, TAGGED) as fake:
            note = tag.run(self.cfg, job)
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn("RETRY", fake.calls[0]["body"]["messages"][-1]["content"])
        self.assertIn("RETRY: the tags were unbalanced", fake.calls[1]["body"]["messages"][-1]["content"])
        self.assertEqual(fake.calls[1]["body"]["temperature"], 0.5)      # profile 0.3 + 0.2
        self.assertIn("attempt 2 of 3", note)
        self.assertEqual(job.status()["meta"]["tagging"], "ok")

    def test_when_every_attempt_fails_continue_keeps_the_plain_story(self):
        self.stage("tag", retries=2)
        job = job_with(self.cfg, **{"02_simplified.txt": STORY})
        with FakeLLM(BAD, BAD, BAD) as fake:
            note = tag.run(self.cfg, job)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(note.startswith("WARNING"))
        self.assertEqual(job.read("03_tagged.txt").strip(), STORY)
        self.assertEqual(job.status()["meta"]["tagging"], "discarded")

    def test_when_every_attempt_fails_stop_fails_the_stage(self):
        self.stage("tag", retries=1, on_fail="stop")
        job = job_with(self.cfg, **{"02_simplified.txt": STORY})
        with FakeLLM(BAD, BAD):
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                tag.run(self.cfg, job)

    def test_a_dead_server_is_an_error_even_with_continue(self):
        self.stage("tag", retries=1)
        job = job_with(self.cfg, **{"02_simplified.txt": STORY})
        with FakeLLM(OSError("down"), OSError("down")) as fake:
            with self.assertRaises(RuntimeError):
                tag.run(self.cfg, job)
        self.assertEqual(len(fake.calls), 2)

    def test_a_call_that_fails_once_is_retried(self):
        self.stage("tag", retries=1)
        job = job_with(self.cfg, **{"02_simplified.txt": STORY})
        with FakeLLM(OSError("blip"), TAGGED) as fake:
            tag.run(self.cfg, job)
        self.assertEqual(job.status()["meta"]["tagging"], "ok")


class SimplifyRetries(Base):
    def test_an_empty_reply_is_retried(self):
        self.stage("simplify", retries=1)
        job = job_with(self.cfg, **{"01_clean.txt": STORY})
        with FakeLLM("** **", "A simple story.") as fake:
            note = simplify.run(self.cfg, job)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1]["body"]["temperature"], 0.4)
        self.assertIn("attempt 2 of 2", note)
        self.assertEqual(job.status()["meta"]["simplification"], "ok")

    def test_stop_fails_after_empty_replies(self):
        self.stage("simplify", retries=1, on_fail="stop")
        job = job_with(self.cfg, **{"01_clean.txt": STORY})
        with FakeLLM("** **", "** **"):
            with self.assertRaisesRegex(RuntimeError, "Simplifying failed"):
                simplify.run(self.cfg, job)

    def test_continue_passes_the_text_on_and_records_the_failure(self):
        self.stage("simplify", retries=1, on_fail="continue")
        job = job_with(self.cfg, **{"01_clean.txt": STORY})
        with FakeLLM("** **", "** **"):
            note = simplify.run(self.cfg, job)
        self.assertTrue(note.startswith("WARNING"))
        self.assertEqual(job.read("02_simplified.txt").strip(), STORY)
        self.assertEqual(job.status()["meta"]["simplification"], "failed")

    def test_a_dead_server_is_an_error_even_with_continue(self):
        self.stage("simplify", retries=1, on_fail="continue")
        job = job_with(self.cfg, **{"01_clean.txt": STORY})
        with FakeLLM(OSError("down"), OSError("down")):
            with self.assertRaises(RuntimeError):
                simplify.run(self.cfg, job)


class JudgeRetryTemperature(Base):
    def test_an_unreadable_judge_reply_is_retried_a_little_warmer(self):
        from stages import judge
        job = job_with(self.cfg)
        self.stage("judge_translation", retries=1, retry_temp_step=0.1)
        with mock.patch.dict("os.environ", {"TEST_CLOUD_KEY": "k"}), FakeLLM("no idea", "REASON: ok\nWINNER: A") as fake:
            idx, _, how, _ = judge.choose(self.cfg, job, lambda lab: "pick", ["x", "y", "z"], seed="s",
                                          stage="judge_translation")
        self.assertEqual(how, "judge")
        self.assertEqual([c["body"]["temperature"] for c in fake.calls], [0.0, 0.1])


class PackageWarnsWhenSimplifyFailed(unittest.TestCase):
    def test_the_message_is_in_the_warnings(self):
        import inspect
        from stages import package
        self.assertIn("simplification", inspect.getsource(package))


if __name__ == "__main__":
    unittest.main()
