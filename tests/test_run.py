"""Step 2 tests: run.py (the command line) and pipeline.py (stage order, skipping, resuming).

Run from the project folder:
    python3 tests/test_run.py -v
    python3 -m unittest discover tests

No model, network or clipboard is used: stdin and the clipboard are replaced, and the pipeline tests
swap in small fake stages. Jobs are created in a temporary folder.
"""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
import run                                         # noqa: E402
from core import job as jobmod                     # noqa: E402
from pipeline import Stage, StageError             # noqa: E402
from tests.helpers import stage_cfg                # noqa: E402

FIX = BASE / "tests" / "fixtures"
NEWSLETTER = (FIX / "messy_synthetic.txt").read_text(encoding="utf-8")


class FakeStdin(io.StringIO):
    def __init__(self, text="", tty=False):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


_REAL_LOAD_RUNNER = pipeline.load_runner


def only_ingest_is_built(stage):
    """The CLI tests are about run.py, not about the later stages, so every stage except ingest counts as
    not built. That keeps these tests true however far the build plan has got."""
    return _REAL_LOAD_RUNNER(stage) if stage.name == "ingest" else None


class CliCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(pipeline, "load_runner", only_ingest_is_built)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)                    # a full, valid config; jobs go in <tmp>/jobs
        self.lines = []

    def call(self, *argv, stdin=None, clipboard=None):
        """Run run.main and return (exit code, standard output lines, standard error text)."""
        err = io.StringIO()
        self.lines = []
        with contextlib.redirect_stderr(err):
            code = run.main(list(argv), cfg=self.cfg, stdin=stdin or FakeStdin(tty=True),
                            clipboard=clipboard, out=self.lines.append)
        return code, "\n".join(self.lines), err.getvalue()

    def write_file(self, name, text):
        path = Path(self.tmp.name) / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def jobs(self):
        return jobmod.list_jobs(self.cfg)


class TestInputRoutes(CliCase):
    def test_text_file_makes_a_job_and_runs_ingest(self):
        path = self.write_file("messy_newsletter.txt", NEWSLETTER)
        code, out, _ = self.call("ingest", "--text", path)
        self.assertEqual(code, 0)
        (job,) = self.jobs()
        self.assertEqual(job.read("00_source.txt"), NEWSLETTER)
        self.assertTrue(job.exists("01_clean.txt"))
        self.assertTrue(job.stage_done("ingest"))
        self.assertIn("messy_newsletter", job.id)
        self.assertIn(f"Started job {job.id}", out)
        self.assertIn(f"Next: python3 run.py simplify --job {job.id}", out)

    def test_pasted_text_is_named_after_its_first_real_words_not_the_boilerplate(self):
        self.call("all", stdin=FakeStdin(NEWSLETTER))
        self.assertIn("THE_WEEKLY_DISPATCH_Issue_42", self.jobs()[0].id)

    def test_asking_for_a_stage_that_is_not_built_is_an_error_but_all_is_not(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        job = self.jobs()[0]
        code, out, _ = self.call("simplify", "--job", job.id)
        self.assertEqual(code, 1)
        self.assertIn("simplify: not built yet", out)
        code, _, _ = self.call("all", "--job", job.id)
        self.assertEqual(code, 0)

    def test_all_three_routes_give_the_same_clean_text(self):
        path = self.write_file("a.txt", NEWSLETTER)
        self.call("all", "--text", path)
        self.call("all", "--text", "-", stdin=FakeStdin(NEWSLETTER))
        self.call("all", stdin=FakeStdin(NEWSLETTER))                      # piped, no flag
        self.call("all", "--clipboard", clipboard=lambda: NEWSLETTER)
        jobs = self.jobs()
        self.assertEqual(len(jobs), 4)
        cleans = {j.read("01_clean.txt") for j in jobs}
        self.assertEqual(len(cleans), 1)
        self.assertEqual({j.read("00_source.txt") for j in jobs}, {NEWSLETTER})

    def test_html_file_is_converted(self):
        path = self.write_file("page.html", (FIX / "messy_synthetic.html").read_text(encoding="utf-8"))
        self.call("ingest", "--text", path)
        self.assertNotIn("<", self.jobs()[0].read("01_clean.txt"))

    def test_a_bom_is_ignored(self):
        path = Path(self.tmp.name) / "bom.txt"
        path.write_bytes(b"\xef\xbb\xbf" + "Hello there, friend.".encode("utf-8"))
        self.call("ingest", "--text", str(path))
        self.assertEqual(self.jobs()[0].read("00_source.txt"), "Hello there, friend.")

    def test_missing_file_is_a_plain_error(self):
        code, _, err = self.call("ingest", "--text", "/no/such/file.txt")
        self.assertEqual(code, 1)
        self.assertIn("File not found", err)
        self.assertEqual(self.jobs(), [])

    def test_not_utf8_is_a_plain_error(self):
        path = Path(self.tmp.name) / "latin.txt"
        path.write_bytes("caf\xe9 au lait".encode("latin-1"))
        code, _, err = self.call("ingest", "--text", str(path))
        self.assertEqual(code, 1)
        self.assertIn("not UTF-8", err)

    def test_empty_input_is_an_error_and_makes_no_job(self):
        for kwargs in ({"stdin": FakeStdin("  \n ")}, {}):
            with self.subTest(kwargs=bool(kwargs)):
                argv = ("all",) if kwargs else ("all", "--text", self.write_file("empty.txt", "\n"))
                code, _, err = self.call(*argv, **kwargs)
                self.assertEqual(code, 1)
                self.assertIn("empty", err)
        self.assertEqual(self.jobs(), [])

    def test_no_source_at_a_terminal_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                run.main(["all"], cfg=self.cfg, stdin=FakeStdin(tty=True), out=lambda s: None)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("say which job", err.getvalue())

    def test_two_sources_are_a_usage_error(self):
        for argv in (["all", "--job", "x", "--text", "f.txt"], ["all", "--latest", "--clipboard"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        run.main(argv, cfg=self.cfg, out=lambda s: None)
                self.assertEqual(caught.exception.code, 2)

    def test_text_only_starts_a_job_with_ingest_or_all(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                run.main(["simplify", "--text", "f.txt"], cfg=self.cfg, out=lambda s: None)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("only start a job with ingest or all", err.getvalue())

    def test_unknown_command_and_from_without_all(self):
        for argv in (["frobnicate", "--latest"], ["ingest", "--latest", "--from", "tag"]):
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        run.main(argv, cfg=self.cfg, out=lambda s: None)
                self.assertEqual(caught.exception.code, 2)

    def test_unknown_job_is_a_plain_error(self):
        code, _, err = self.call("ingest", "--job", "nope")
        self.assertEqual(code, 1)
        self.assertIn("No job named nope", err)

    def test_clipboard_failure_is_a_plain_error(self):
        with mock.patch("run.subprocess.run", side_effect=OSError("no pbpaste")):
            code, _, err = self.call("ingest", "--clipboard", clipboard=run.read_clipboard)
        self.assertEqual(code, 1)
        self.assertIn("pbpaste", err)


class TestRerunsAndStatus(CliCase):
    def test_a_finished_stage_is_skipped_on_a_rerun(self):
        path = self.write_file("a.txt", NEWSLETTER)
        self.call("ingest", "--text", path)
        job = self.jobs()[0]
        with mock.patch("stages.ingest.run") as spy:
            code, out, _ = self.call("ingest", "--job", job.id)
        self.assertEqual(code, 0)
        spy.assert_not_called()
        self.assertIn("already done", out)

    def test_force_runs_it_again(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        job = self.jobs()[0]
        with mock.patch("stages.ingest.run", return_value="again") as spy:
            self.call("ingest", "--job", job.id, "--force")
        spy.assert_called_once()

    def test_a_missing_output_file_means_the_stage_runs_again(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        job = self.jobs()[0]
        job.file("01_clean.txt").unlink()
        code, out, _ = self.call("ingest", "--job", job.id)
        self.assertEqual(code, 0)
        self.assertTrue(job.exists("01_clean.txt"))
        self.assertIn("ingest: done", out)

    def test_all_stops_at_the_first_stage_that_is_not_built(self):
        code, out, _ = self.call("all", "--text", self.write_file("a.txt", NEWSLETTER))
        self.assertEqual(code, 0)
        job = self.jobs()[0]
        self.assertTrue(job.stage_done("ingest"))
        self.assertIn("simplify: not built yet", out)
        self.assertIn("Stopped: the 'simplify' stage is not built yet.", out)

    def test_latest_picks_the_newest_job(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        self.call("ingest", "--text", self.write_file("b.txt", "Second article text here."))
        code, out, _ = self.call("status", "--latest")
        self.assertEqual(code, 0)
        self.assertIn(self.jobs()[-1].id, out)
        self.assertIn("ingest", out)

    def test_status_and_list(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        job = self.jobs()[0]
        _, out, _ = self.call("status", "--job", job.id)
        self.assertRegex(out, r"ingest\s+done")
        self.assertRegex(out, r"simplify\s+pending")
        self.assertEqual(len([l for l in self.lines if l.strip().startswith(("ingest", "simplify", "tag", "translate",
                                                                           "judge", "count", "revert", "package"))]), 9)
        _, out, _ = self.call("list")
        self.assertIn(f"{job.id}   next: simplify", out)

    def test_list_with_no_jobs(self):
        _, out, _ = self.call("list")
        self.assertEqual(out, "No jobs yet.")

    def test_a_bad_stage_for_the_job_names_the_missing_file(self):
        self.call("ingest", "--text", self.write_file("a.txt", NEWSLETTER))
        job = self.jobs()[0]
        with mock.patch.object(pipeline, "load_runner", lambda stage: (lambda cfg, job: "x")):
            job.file("01_clean.txt").unlink()
            code, _, err = self.call("simplify", "--job", job.id)
        self.assertEqual(code, 1)
        self.assertIn("simplify needs 01_clean.txt", err)
        self.assertIn(f"python3 run.py ingest --job {job.id}", err)

    def test_ingest_failure_is_marked_in_the_job_and_logged(self):
        code, _, err = self.call("ingest", "--text", self.write_file("junk.txt", "Unsubscribe\n\nView in browser\n"))
        self.assertEqual(code, 1)
        job = self.jobs()[0]
        self.assertIn("ingest failed: Nothing was left", err)
        self.assertEqual(job.status()["stages"]["ingest"]["status"], "failed")
        self.assertRegex(next(l for l in pipeline.job_report(job) if "ingest" in l), r"ingest\s+failed")


# ------------------------------------------------------------------ pipeline logic, with fake stages

def fake_module(name, calls, writes, note="ok", fail=False):
    mod = types.ModuleType(name)

    def run_(cfg, job):
        calls.append(name)
        if fail:
            raise ValueError("boom")
        for f in writes:
            job.write(f, f"{name} output")
        return note
    mod.run = run_
    return mod


class TestPipelineLogic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {"paths": {"jobs": self.tmp.name}}
        self.job = jobmod.new_job(self.cfg, "source text")
        self.calls = []
        self.lines = []
        self.flag = {"revert": False}
        self.mods = {
            "fake.a": fake_module("fake.a", self.calls, ["a.txt"]),
            "fake.b": fake_module("fake.b", self.calls, ["b.txt"]),
            "fake.rev": fake_module("fake.rev", self.calls, ["07_x.txt"]),
            "fake.c": fake_module("fake.c", self.calls, ["c.txt"]),
        }
        stages = [
            Stage("a", "fake.a", needs=("00_source.txt",), makes=("a.txt",)),
            Stage("b", "fake.b", needs=("a.txt",), makes=("b.txt",)),
            Stage("rev", "fake.rev", needs=("b.txt",), makes=("07_x.txt",), when=lambda j: self.flag["revert"],
                  cleans=("07_*",)),
            Stage("c", "fake.c", needs=("b.txt",), makes=("c.txt",)),
            Stage("later", "fake.missing", needs=("c.txt",), makes=("l.txt",)),
        ]
        for patcher in (mock.patch.object(pipeline, "STAGES", stages),
                        mock.patch.dict(sys.modules, self.mods)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_all(self, **kw):
        self.lines = []
        return pipeline.run_pipeline(self.cfg, self.job, out=self.lines.append, **kw)

    def test_runs_in_order_and_stops_at_a_stage_that_is_not_built(self):
        results = self.run_all()
        self.assertEqual(self.calls, ["fake.a", "fake.b", "fake.c"])
        self.assertEqual(results, [("a", "done"), ("b", "done"), ("rev", "not_needed"),
                                   ("c", "done"), ("later", "not_built")])

    def test_done_stages_are_skipped_on_the_second_run(self):
        self.run_all()
        self.calls.clear()
        self.run_all()
        self.assertEqual(self.calls, [])

    def test_resuming_after_a_failure_reruns_only_the_failed_stage(self):
        self.mods["fake.b"] = fake_module("fake.b", self.calls, ["b.txt"], fail=True)
        sys.modules["fake.b"] = self.mods["fake.b"]
        with self.assertRaises(StageError) as caught:
            self.run_all()
        self.assertIn("b failed: boom", str(caught.exception))
        self.assertEqual(self.job.status()["stages"]["b"]["status"], "failed")
        self.assertIn("Traceback", self.job.read("log.txt"))
        self.calls.clear()
        sys.modules["fake.b"] = fake_module("fake.b", self.calls, ["b.txt"])
        self.run_all()
        self.assertEqual(self.calls, ["fake.b", "fake.c"])

    def test_from_reruns_that_stage_and_everything_after_it(self):
        self.run_all()
        self.calls.clear()
        self.run_all(start="b")
        self.assertEqual(self.calls, ["fake.b", "fake.c"])

    def test_forcing_a_stage_forgets_the_later_ones(self):
        self.run_all()
        pipeline.run_stage(self.cfg, self.job, "a", force=True, out=lambda s: None)
        stages = self.job.status()["stages"]
        self.assertEqual(list(stages), ["a"])
        self.assertEqual(self.job.status()["last_finished"], "a")

    def test_a_stage_that_does_not_write_its_output_is_an_error(self):
        sys.modules["fake.a"] = fake_module("fake.a", self.calls, [])
        with self.assertRaisesRegex(StageError, "did not write a.txt"):
            self.run_all()

    def test_a_stage_needing_a_missing_file_says_which_stage_makes_it(self):
        with self.assertRaisesRegex(StageError, r"c needs b.txt.*python3 run.py b --job"):
            pipeline.run_stage(self.cfg, self.job, "c", out=lambda s: None)

    def test_a_conditional_stage_runs_when_its_condition_is_true(self):
        self.flag["revert"] = True
        self.assertIn(("rev", "done"), self.run_all())

    def test_a_skipped_conditional_stage_removes_its_stale_files(self):
        self.flag["revert"] = True
        self.run_all()
        self.assertTrue(self.job.exists("07_x.txt"))
        self.flag["revert"] = False
        self.run_all(start="b")
        self.assertFalse(self.job.exists("07_x.txt"))
        self.assertEqual(self.job.status()["stages"]["rev"]["status"], "skipped")

    def test_next_stage_ignores_skipped_stages(self):
        self.run_all()
        self.assertEqual(pipeline.next_stage(self.job), "later")

    def test_unknown_stage_names_the_real_ones(self):
        with self.assertRaisesRegex(Exception, "Unknown stage 'nope'. Stages, in order: a, b, rev, c, later"):
            pipeline.get_stage("nope")


class TestRealRegistry(unittest.TestCase):
    def test_every_stage_needs_what_an_earlier_stage_makes(self):
        made = {"00_source.txt"}
        for stage in pipeline.STAGES:
            for f in stage.needs:
                self.assertIn(f, made, f"{stage.name} needs {f}, which no earlier stage makes")
            made.update(stage.makes)

    def test_stage_names_are_unique_and_in_the_planned_order(self):
        self.assertEqual(pipeline.names(), ["ingest", "simplify", "tag", "translate", "judge_translation", "count",
                                            "revert", "judge_comprehension", "package", "mail_out"])

    def test_ingest_is_built(self):
        self.assertIsNotNone(_REAL_LOAD_RUNNER(pipeline.get_stage("ingest")))

    def test_the_job_file_stays_valid_json_after_every_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"paths": {"jobs": tmp}}
            job = jobmod.new_job(cfg, "Some article text for the test.")
            pipeline.run_stage(cfg, job, "ingest", out=lambda s: None)
            json.loads(job.file("job.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
