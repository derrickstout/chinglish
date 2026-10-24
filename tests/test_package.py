"""Step 8 tests: stages/package.py, and the whole pipeline run end to end with scripted fake models.

Run from the project folder:
    python3 tests/test_package.py -v
    python3 -m unittest discover tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
import run                                         # noqa: E402
from stages import count, package                  # noqa: E402
from tests.fakes import FakeLLM                    # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

FIX = BASE / "tests" / "fixtures"
ALLOWED = json.loads((FIX / "allowed_chars_sample.json").read_text(encoding="utf-8"))
KNOWN = ALLOWED["mastered"] + ALLOWED["learning"]
POOL = [u["char"] for u in ALLOWED["unknown"]]
RARE = [chr(c) for c in range(0x9F00, 0x9F00 + 120) if chr(c) not in KNOWN and chr(c) not in POOL]

SOURCE = "Dana walked to the market and bought apples near the river."
TAGGED = "<dnt>Dana</dnt> walked to the market and bought apples near the river."
OK_TEXT = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:5]) + "。"                 # 5 new: within target
BIG_TEXT = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:70]) + "。"              # 70 new: needs reversion
REVERTED = "<dnt>Dana</dnt>" + "".join(KNOWN[:6]) + "".join(RARE[:20]) + "<en>market</en>。"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        root = Path(self.tmp.name)
        (root / "data").mkdir()
        shutil.copy(FIX / "allowed_chars_sample.json", root / "data" / "allowed_chars.json")
        for f in (BASE / "prompts").glob("*.txt"):                 # the real prompts, except the two the helper wrote
            if f.name not in ("simplify.txt", "dnt.txt"):
                shutil.copy(f, Path(self.cfg["paths"]["prompts"]) / f.name)

    def job(self, winner, **extra):
        files = {"01_clean.txt": SOURCE + "\n", "02_simplified.txt": SOURCE + " (simple)\n",
                 "03_tagged.txt": TAGGED + "\n", "05_winner.txt": winner + "\n"}
        files.update(extra)
        job = job_with(self.cfg, **files)
        job.set_meta(tagging="ok")
        job.write_json("05_judge.json", {"winner": 2, "temperature": 0.6, "method": "judge",
                                         "reason": "most natural", "candidates": [1, 2, 3]})
        job.write_json("04_meta.json", {"candidates": [{"n": 2, "temperature": 0.6, "file": "x", "notes": "a note"}]})
        count.run(self.cfg, job)
        return job


class PackageText(Base):
    def test_the_result_has_the_four_parts_with_tags_removed(self):
        job = self.job(OK_TEXT)
        package.run(self.cfg, job)
        text = job.read("09_result.txt")
        for heading in ("TRANSLATION", "ENGLISH (simplified)", "ORIGINAL ARTICLE", "NOTES"):
            self.assertIn(heading, text)
        self.assertNotIn("<dnt>", text)
        self.assertIn("Dana" + "".join(KNOWN[:6]), text)
        self.assertIn(SOURCE + " (simple)", text)
        self.assertLess(text.index("TRANSLATION"), text.index("ENGLISH (simplified)"))
        self.assertLess(text.index("ORIGINAL ARTICLE"), text.index("NOTES"))

    def test_a_translation_within_the_target_is_packaged_from_05(self):
        job = self.job(OK_TEXT)
        note = package.run(self.cfg, job)
        data = job.read_json("09_result.json")
        self.assertEqual((data["source"], data["kind"], data["new_chars"]), ("05_winner.txt", "translation", 5))
        self.assertIn("packaged 05_winner.txt (5 new characters, target 50)", note)
        self.assertNotIn("WARNING", note)

    def test_the_notes_say_how_the_translation_was_chosen_and_where_the_count_stands(self):
        job = self.job(OK_TEXT)
        package.run(self.cfg, job)
        notes = "\n".join(job.read_json("09_result.json")["notes"])
        self.assertIn("Translation 2 of 3 (temperature 0.6), chosen by the cloud judge. Judge: most natural", notes)
        self.assertIn("New characters in this text: 5 (target 50); within the target.", notes)
        self.assertIn("The model's note on the translation: a note", notes)
        self.assertIn("preferred list", notes + "outside your preferred list")

    def test_the_new_characters_left_are_listed(self):
        job = self.job(OK_TEXT)
        package.run(self.cfg, job)
        notes = "\n".join(job.read_json("09_result.json")["notes"])
        self.assertIn("outside your preferred list (5): " + " ".join(RARE[:5]), notes)

    def test_english_put_back_stays_as_plain_words(self):
        job = self.job(BIG_TEXT, **{"08_winner.txt": REVERTED + "\n", "07_meta.json": "{}"})
        job.write_json("07_meta.json", {"candidates": [{"n": 1, "temperature": 0.3, "en_spans": 1, "valid": True}]})
        job.write_json("08_judge.json", {"winner": 1, "method": "judge", "reason": "flows"})
        package.run(self.cfg, job)
        data = job.read_json("09_result.json")
        self.assertIn("market", data["translation"])
        self.assertNotIn("<en>", data["translation"])
        self.assertEqual((data["source"], data["kind"]), ("08_winner.txt", "reversion"))

    def test_reversion_is_explained_in_the_notes(self):
        job = self.job(BIG_TEXT, **{"08_winner.txt": REVERTED + "\n"})
        job.write_json("07_meta.json", {"candidates": [{"n": 1, "temperature": 0.3, "en_spans": 1, "valid": True}]})
        job.write_json("08_judge.json", {"winner": 1, "method": "judge", "reason": "flows"})
        package.run(self.cfg, job)
        notes = "\n".join(job.read_json("09_result.json")["notes"])
        self.assertIn("English reversion ran: 70 new characters before, 20 after, with 1 English insertion", notes)

    def test_a_stale_08_file_is_ignored_when_reversion_is_not_needed(self):
        job = self.job(OK_TEXT, **{"08_winner.txt": "STALE\n"})
        package.run(self.cfg, job)
        self.assertNotIn("STALE", job.read("09_result.txt"))

    def test_reversion_that_produced_nothing_says_so_and_warns(self):
        job = self.job(BIG_TEXT)                                   # needed, but there is no 08_winner.txt
        note = package.run(self.cfg, job)
        self.assertIn("WARNING", note)
        notes = "\n".join(job.read_json("09_result.json")["notes"])
        self.assertIn("no attempt passed the checks", notes)

    def test_still_over_the_target_after_everything_is_a_warning(self):
        job = self.job(BIG_TEXT, **{"08_winner.txt": BIG_TEXT + "\n"})
        job.write_json("07_meta.json", {"candidates": []})
        job.write_json("08_judge.json", {"winner": None, "method": "kept_translation"})
        note = package.run(self.cfg, job)
        self.assertIn("still 20 over the new-character target of 50", note)

    def test_the_warnings_come_first_in_the_notes(self):
        job = self.job(BIG_TEXT)
        package.run(self.cfg, job)
        self.assertEqual(job.read_json("09_result.json")["notes"][0], "WARNINGS:")


class WarningsFromEarlierStages(Base):
    def notes(self, job):
        package.run(self.cfg, job)
        return "\n".join(job.read_json("09_result.json")["notes"])

    def test_a_cut_article_is_reported(self):
        job = self.job(OK_TEXT)
        job.set_meta(ingest_cut_to=12000)
        self.assertIn("only the beginning was translated (the first 12,000 characters)", self.notes(job))

    def test_an_old_job_with_only_the_ingest_note_is_still_reported(self):
        job = self.job(OK_TEXT)
        data = job.status()
        data.setdefault("stages", {})["ingest"] = {"status": "done", "note": "5 -> 4 characters; CUT TO THE FIRST 9 CHARACTERS"}
        job.write_json("job.json", data)
        self.assertIn("only the beginning was translated", self.notes(job))

    def test_discarded_tagging_is_reported(self):
        job = self.job(OK_TEXT)
        job.set_meta(tagging="discarded", tagging_problem="the tags were unbalanced")
        self.assertIn("names could not be protected", self.notes(job))

    def test_a_judge_fallback_is_reported(self):
        job = self.job(OK_TEXT)
        j = job.read_json("05_judge.json")
        j["method"] = "fallback"
        job.write_json("05_judge.json", j)
        self.assertIn("answer could not be read", self.notes(job))

    def test_an_old_character_list_gets_a_refresh_hint(self):
        job = self.job(OK_TEXT)
        path = Path(self.tmp.name) / "data" / "allowed_chars.json"
        old = path.stat().st_mtime - 90 * 86400
        os.utime(path, (old, old))
        self.assertIn("Re-export from Pleco", self.notes(job))

    def test_a_fresh_character_list_gets_no_hint(self):
        self.assertNotIn("Re-export", self.notes(self.job(OK_TEXT)))

    def test_a_lost_protected_name_is_reported(self):
        job = self.job("".join(KNOWN[:6]) + "。")
        self.assertIn("protected text was missing", self.notes(job))


class EndToEnd(Base):
    """run.py all, with the local and cloud models both scripted."""

    def script(self, simplified, tagged, translations, reversions):
        local = [simplified, tagged] + translations + reversions
        cloud = ["REASON: fine.\nWINNER: A"] * 2

        def handler(body, i):
            queue = cloud if body["model"] == "judge-model" else local
            return queue.pop(0)
        return handler

    def run_all(self, handler):
        src = Path(self.tmp.name) / "article.txt"
        src.write_text(SOURCE + "\n", encoding="utf-8")
        out = []
        env = mock.patch.dict(os.environ, {"TEST_CLOUD_KEY": "k"})
        with env, FakeLLM(handler=handler):
            code = run.main(["all", "--text", str(src)], cfg=self.cfg, out=out.append)
        return code, out

    def job_dir(self):
        return next((Path(self.tmp.name) / "jobs").iterdir())

    def test_a_short_translation_runs_every_stage_but_reversion(self):
        handler = self.script(SOURCE, TAGGED, [OK_TEXT] * 3, [])
        code, out = self.run_all(handler)
        self.assertEqual(code, 0)
        text = "\n".join(out)
        self.assertIn("revert: not needed", text)
        self.assertIn("judge_comprehension: not needed", text)
        self.assertIn("package: done", text)
        self.assertIn("Result: ", text)
        self.assertNotIn("Next:", text)
        self.assertTrue((self.job_dir() / "09_result.txt").exists())
        self.assertFalse(list(self.job_dir().glob("07_*")))

    def test_a_long_translation_also_runs_reversion_and_the_second_judge(self):
        handler = self.script(SOURCE, TAGGED, [BIG_TEXT] * 3, [REVERTED, REVERTED, REVERTED])
        code, out = self.run_all(handler)
        self.assertEqual(code, 0)
        data = json.loads((self.job_dir() / "09_result.json").read_text(encoding="utf-8"))
        self.assertEqual((data["kind"], data["new_chars"]), ("reversion", 20))
        self.assertIn("revert: done", "\n".join(out))

    def test_every_stage_is_marked_done_or_skipped_and_the_job_is_complete(self):
        handler = self.script(SOURCE, TAGGED, [OK_TEXT] * 3, [])
        self.run_all(handler)
        from core import job as jobmod
        job = jobmod.open_job(self.cfg, self.job_dir().name)
        self.assertIsNone(pipeline.next_stage(job))

    def test_running_all_again_skips_everything_and_makes_no_model_call(self):
        handler = self.script(SOURCE, TAGGED, [OK_TEXT] * 3, [])
        self.run_all(handler)
        out = []
        with FakeLLM() as fake:
            code = run.main(["all", "--latest"], cfg=self.cfg, out=out.append)
        self.assertEqual((code, fake.calls), (0, []))
        self.assertIn("package: already done", "\n".join(out))


if __name__ == "__main__":
    unittest.main()
