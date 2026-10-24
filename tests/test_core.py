"""Step 1 tests: core/config.py, core/job.py, core/llm.py, core/textcheck.py.

Run from the project folder:
    python3 tests/test_core.py -v
    python3 -m unittest discover tests          # every test in tests/

No LM Studio or cloud key is needed: model calls go to the scripted fake in tests/fakes.py.

Regression against the old check_text.py (optional): put the old file anywhere and run
    OLD_CHECK_TEXT=/path/to/check_text.py python3 tests/test_core.py -v TestRegression
The new core/textcheck.py must give the same counts, problems and warnings as the old script.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import config, job as jobmod, llm, textcheck          # noqa: E402
from core.config import ConfigError                              # noqa: E402
from tests.fakes import FakeLLM, http_error, reply_json          # noqa: E402

FIX = BASE / "tests" / "fixtures"


def read_fixture(name):
    return (FIX / name).read_text(encoding="utf-8")


def sample_cfg(jobs_dir="jobs"):
    return {
        "paths": {"jobs": jobs_dir, "data": "data"},
        "models": {
            "local1": {"url": "http://fake:1234/v1", "model": "fake-model", "server": "lm_studio",
                       "temperature": 0.3, "max_tokens": 4096, "ttl": 600},
            "cloud1": {"url": "http://fake-cloud/v1", "model": "judge-model", "api_key_env": "TEST_CLOUD_KEY",
                       "temperature": 0.0, "max_tokens": 2048},
        },
        "translation": {"target_new_chars": 50, "revert_tolerance": 10},
    }


# ---------------------------------------------------------------- config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        config._last_good = {}

    def test_load_and_reread_after_an_edit(self):
        self.path.write_text(json.dumps({"models": {"m": {"model": "a"}}}), encoding="utf-8")
        self.assertEqual(config.get(config.current_config(self.path), "models.m.model"), "a")
        self.path.write_text(json.dumps({"models": {"m": {"model": "b"}}}), encoding="utf-8")
        self.assertEqual(config.get(config.current_config(self.path), "models.m.model"), "b")

    def test_half_saved_file_keeps_the_last_good_copy(self):
        self.path.write_text(json.dumps({"x": 1}), encoding="utf-8")
        config.current_config(self.path)
        self.path.write_text('{"x": 2, "y"', encoding="utf-8")            # cut off mid-save
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cfg = config.current_config(self.path)
        self.assertEqual(cfg["x"], 1)
        self.assertIn("previous settings", out.getvalue())

    def test_missing_file_with_no_earlier_copy_raises(self):
        with self.assertRaises(ConfigError) as cm:
            config.current_config(self.path)
        self.assertIn("config.json", str(cm.exception))

    def test_invalid_json_raises_a_plain_message(self):
        self.path.write_text("{nope", encoding="utf-8")
        with self.assertRaises(ConfigError) as cm:
            config.load_config(self.path)
        self.assertIn("not valid JSON", str(cm.exception))

    def test_require_names_the_missing_key(self):
        cfg = {"models": {"m": {"model": ""}}}
        with self.assertRaises(ConfigError) as cm:
            config.require(cfg, "models.m.model")
        self.assertIn('"models.m.model"', str(cm.exception))
        with self.assertRaises(ConfigError):
            config.require(cfg, "models.other.url")

    def test_get_with_dotted_keys_and_defaults(self):
        cfg = {"a": {"b": {"c": 5}}}
        self.assertEqual(config.get(cfg, "a.b.c"), 5)
        self.assertEqual(config.get(cfg, "a.x.c", "dflt"), "dflt")
        self.assertIsNone(config.get(cfg, "z"))

    def test_resolve_path_is_under_the_project_folder_unless_absolute(self):
        self.assertEqual(config.resolve_path({"paths": {"jobs": "jobs"}}, "jobs"), config.BASE / "jobs")
        self.assertEqual(config.resolve_path({"paths": {"jobs": "/tmp/j"}}, "jobs"), Path("/tmp/j"))
        with self.assertRaises(ConfigError):
            config.resolve_path({"paths": {}}, "jobs")


# ---------------------------------------------------------------- job folders

class TestJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = sample_cfg(str(Path(self.tmp.name) / "jobs"))

    def test_new_job_holds_the_source_and_a_status_file(self):
        j = jobmod.new_job(self.cfg, "The fox and the grapes, a short story.")
        self.assertTrue(j.exists("00_source.txt"))
        self.assertEqual(j.read("00_source.txt"), "The fox and the grapes, a short story.")
        self.assertEqual(j.status()["stages"], {})
        self.assertIn("The_fox_and_the_grapes", j.id)

    def test_files_round_trip_with_chinese_text(self):
        j = jobmod.new_job(self.cfg, "x")
        j.write("05_winner.txt", "你好，世界。\n")
        self.assertEqual(j.read("05_winner.txt"), "你好，世界。\n")
        j.write_json("06_count.json", {"new": 3, "chars": "貓狗"})
        self.assertEqual(j.read_json("06_count.json"), {"new": 3, "chars": "貓狗"})

    def test_writes_leave_no_temporary_file_behind(self):
        j = jobmod.new_job(self.cfg, "x")
        j.write("a.txt", "one")
        j.write("a.txt", "two")
        self.assertEqual(j.read("a.txt"), "two")
        self.assertEqual([p.name for p in j.path.glob("*.tmp")], [])

    def test_reading_a_missing_file_says_which_one(self):
        j = jobmod.new_job(self.cfg, "x")
        with self.assertRaises(FileNotFoundError) as cm:
            j.read("03_tagged.txt")
        self.assertIn("03_tagged.txt", str(cm.exception))

    def test_stage_status_survives_reopening_the_job(self):
        j = jobmod.new_job(self.cfg, "x")
        self.assertFalse(j.stage_done("ingest"))
        j.mark_stage("ingest", seconds=0.42)
        j.mark_stage("simplify", seconds=12.34, note="ok")
        again = jobmod.open_job(self.cfg, j.id)
        self.assertTrue(again.stage_done("ingest"))
        self.assertEqual(again.done_stages(), ["ingest", "simplify"])
        self.assertEqual(again.status()["last_finished"], "simplify")
        self.assertEqual(again.status()["stages"]["simplify"]["seconds"], 12.3)

    def test_a_failed_stage_is_not_done(self):
        j = jobmod.new_job(self.cfg, "x")
        j.mark_failed("translate", "model hit the token limit")
        self.assertFalse(j.stage_done("translate"))
        self.assertEqual(j.status()["stages"]["translate"]["error"], "model hit the token limit")

    def test_meta_is_kept_alongside_stages(self):
        j = jobmod.new_job(self.cfg, "x", subject="Daily brief")
        j.set_meta(message_id="<abc@mail>")
        j.mark_stage("ingest")
        meta = j.status()["meta"]
        self.assertEqual(meta, {"subject": "Daily brief", "message_id": "<abc@mail>"})

    def test_same_name_never_overwrites_an_earlier_job(self):
        a = jobmod.new_job(self.cfg, "Same text here")
        b = jobmod.new_job(self.cfg, "Same text here")
        self.assertNotEqual(a.id, b.id)
        self.assertTrue(b.id.endswith("_2"))

    def test_list_and_latest_come_oldest_first(self):
        self.assertEqual(jobmod.list_jobs(self.cfg), [])
        self.assertIsNone(jobmod.latest_job(self.cfg))
        a = jobmod.new_job(self.cfg, "aaa", slug="a")
        b = jobmod.new_job(self.cfg, "bbb", slug="b")
        self.assertEqual([j.id for j in jobmod.list_jobs(self.cfg)], sorted([a.id, b.id]))
        self.assertEqual(jobmod.latest_job(self.cfg).id, sorted([a.id, b.id])[-1])

    def test_opening_an_unknown_job_gives_a_plain_message(self):
        with self.assertRaises(ConfigError) as cm:
            jobmod.open_job(self.cfg, "nope")
        self.assertIn("No job named nope", str(cm.exception))

    def test_slug_handles_empty_and_non_latin_text(self):
        self.assertEqual(jobmod.make_slug("你好"), "story")
        self.assertEqual(jobmod.make_slug("Hello, wonderful world!"), "Hello_wonderful_world")


# ---------------------------------------------------------------- model calls (fake server)

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.cfg = sample_cfg()
        quiet = contextlib.redirect_stdout(io.StringIO())          # hide the "model wrote N tokens" lines
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_prompt_and_text_become_system_and_user_messages(self):
        with FakeLLM("ok") as fake:
            self.assertEqual(llm.chat(self.cfg, "Be brief.", "Hello", model="local1"), "ok")
        body = fake.calls[0]["body"]
        self.assertEqual(body["messages"], [{"role": "system", "content": "Be brief."},
                                            {"role": "user", "content": "Hello"}])
        self.assertEqual(fake.calls[0]["url"], "http://fake:1234/v1/chat/completions")
        self.assertEqual(body["model"], "fake-model")

    def test_prompt_alone_is_one_user_message(self):
        with FakeLLM("ok") as fake:
            llm.chat(self.cfg, "Just this", model="local1")
        self.assertEqual(fake.calls[0]["body"]["messages"], [{"role": "user", "content": "Just this"}])

    def test_a_whole_conversation_can_be_sent(self):
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                 {"role": "user", "content": "again"}]
        with FakeLLM("ok") as fake:
            llm.chat(self.cfg, messages=convo, model="local1")
        self.assertEqual(fake.calls[0]["body"]["messages"], convo)

    def test_user_suffix_goes_on_the_last_user_message_only(self):
        self.cfg["models"]["local1"]["user_suffix"] = "/no_think"
        convo = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
                 {"role": "user", "content": "second"}]
        with FakeLLM("ok", "ok") as fake:
            llm.chat(self.cfg, messages=convo, model="local1")
            llm.chat(self.cfg, "sys", "story", model="local1")
        sent = fake.calls[0]["body"]["messages"]
        self.assertEqual((sent[0]["content"], sent[2]["content"]), ("first", "second\n/no_think"))
        self.assertEqual(fake.calls[1]["body"]["messages"][1]["content"], "story\n/no_think")
        self.assertEqual(convo[2]["content"], "second")                 # the caller's list is not changed

    def test_default_and_overridden_temperature(self):
        with FakeLLM("a", "b") as fake:
            llm.chat(self.cfg, "p", model="local1")
            llm.chat(self.cfg, "p", temperature=0.6, model="local1")
        self.assertEqual(fake.calls[0]["body"]["temperature"], 0.3)
        self.assertEqual(fake.calls[1]["body"]["temperature"], 0.6)

    def test_temperature_zero_is_honoured(self):
        with FakeLLM("a") as fake:
            llm.chat(self.cfg, "p", temperature=0, model="local1")
        self.assertEqual(fake.calls[0]["body"]["temperature"], 0)

    def test_extra_settings_are_sent_but_an_explicit_temperature_wins(self):
        self.cfg["models"]["local1"]["extra"] = {"top_p": 0.9, "temperature": 0.1}
        with FakeLLM("a", "b") as fake:
            llm.chat(self.cfg, "p", model="local1")
            llm.chat(self.cfg, "p", temperature=0.8, model="local1")
        self.assertEqual(fake.calls[0]["body"]["top_p"], 0.9)
        self.assertEqual(fake.calls[0]["body"]["temperature"], 0.1)
        self.assertEqual(fake.calls[1]["body"]["temperature"], 0.8)

    def test_think_blocks_are_stripped(self):
        with FakeLLM("<think>hmm,\nlet me see</think>\n你好。") as fake:
            self.assertEqual(llm.chat(self.cfg, "p", model="local1"), "你好。")

    def test_max_tokens_argument_beats_the_setting(self):
        with FakeLLM("a", "b") as fake:
            llm.chat(self.cfg, "p", model="local1")
            llm.chat(self.cfg, "p", max_tokens=123, model="local1")
        self.assertEqual(fake.calls[0]["body"]["max_tokens"], 4096)
        self.assertEqual(fake.calls[1]["body"]["max_tokens"], 123)

    def test_empty_reply_raises(self):
        with FakeLLM("<think>only thinking</think>") as fake:
            with self.assertRaises(RuntimeError) as cm:
                llm.chat(self.cfg, "p", model="local1")
        self.assertIn("no text", str(cm.exception))

    def test_empty_reply_after_hitting_the_token_limit_says_so(self):
        with FakeLLM(reply_json("", finish="length")) as fake:
            with self.assertRaises(RuntimeError) as cm:
                llm.chat(self.cfg, "p", model="local1")
        self.assertIn("max_tokens", str(cm.exception))

    def test_local_calls_send_ttl_and_reasoning_effort(self):
        self.cfg["models"]["local1"]["reasoning_effort"] = "none"
        with FakeLLM("ok") as fake:
            llm.chat(self.cfg, "p", model="local1")
        body = fake.calls[0]["body"]
        self.assertEqual((body["ttl"], body["reasoning_effort"]), (600, "none"))
        self.assertNotIn("Authorization", fake.calls[0]["headers"])

    def test_cloud_calls_use_the_cloud_model_and_key_and_no_lm_studio_fields(self):
        self.cfg["models"]["local1"]["reasoning_effort"] = "none"
        with mock.patch.dict(os.environ, {"TEST_CLOUD_KEY": "secret-123"}):
            with FakeLLM("WINNER: B") as fake:
                reply = llm.chat(self.cfg, "judge this", model="cloud1")
        self.assertEqual(reply, "WINNER: B")
        call = fake.calls[0]
        self.assertEqual(call["url"], "http://fake-cloud/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-123")
        self.assertEqual(call["body"]["model"], "judge-model")
        self.assertEqual(call["body"]["temperature"], 0.0)
        self.assertNotIn("ttl", call["body"])
        self.assertNotIn("reasoning_effort", call["body"])

    def test_cloud_call_without_a_key_explains_how_to_set_it(self):
        env = {k: v for k, v in os.environ.items() if k != "TEST_CLOUD_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with FakeLLM("x") as fake:
                with self.assertRaises(ConfigError) as cm:
                    llm.chat(self.cfg, "p", model="cloud1")
        self.assertIn("export TEST_CLOUD_KEY", str(cm.exception))
        self.assertEqual(fake.calls, [])

    def test_missing_model_names_the_setting(self):
        self.cfg["models"]["local1"]["model"] = ""
        with self.assertRaises(ConfigError) as cm:
            llm.chat(self.cfg, "p", model="local1")
        self.assertIn('"models.local1.model"', str(cm.exception))

    def test_an_undefined_profile_lists_the_defined_ones(self):
        with FakeLLM("x") as fake:
            with self.assertRaisesRegex(ConfigError, 'model "nope" is not defined.*local1, cloud1'):
                llm.chat(self.cfg, "p", model="nope")
        self.assertEqual(fake.calls, [])

    def test_no_model_name_is_a_plain_error(self):
        with self.assertRaisesRegex(ConfigError, "stages.<stage>.model"):
            llm.chat(self.cfg, "p")

    def test_any_number_of_profiles_can_be_used_by_name(self):
        self.cfg["models"]["local7"] = dict(self.cfg["models"]["local1"], model="seventh", url="http://other:9/v1")
        with FakeLLM("a") as fake:
            llm.chat(self.cfg, "p", model="local7")
        self.assertEqual((fake.calls[0]["url"], fake.calls[0]["body"]["model"]),
                         ("http://other:9/v1/chat/completions", "seventh"))

    def test_a_profile_without_server_lm_studio_gets_no_lm_studio_fields(self):
        self.cfg["models"]["local1"].update(server="", reasoning_effort="none")
        with FakeLLM("a") as fake:
            llm.chat(self.cfg, "p", model="local1")
        self.assertNotIn("ttl", fake.calls[0]["body"])
        self.assertNotIn("reasoning_effort", fake.calls[0]["body"])

    def test_model_switch_does_nothing_for_a_profile_that_is_not_lm_studio(self):
        llm.LOADED_MODEL["name"] = "old"
        with mock.patch("core.llm.subprocess.run") as run:
            llm.ensure_model_switched(self.cfg, "cloud1")
            run.assert_not_called()
        self.assertEqual(llm.LOADED_MODEL["name"], "old")

    def test_http_error_becomes_a_readable_message(self):
        with FakeLLM(http_error(500, "model crashed")) as fake:
            with self.assertRaises(RuntimeError) as cm:
                llm.chat(self.cfg, "p", model="local1")
        self.assertIn("model crashed", str(cm.exception))

    def test_unreachable_server_becomes_a_readable_message(self):
        with FakeLLM(urllib.error.URLError("connection refused")) as fake:
            with self.assertRaises(RuntimeError) as cm:
                llm.chat(self.cfg, "p", model="local1")
        self.assertIn("Could not reach", str(cm.exception))

    def test_list_models(self):
        with FakeLLM(models=["a", "b"]):
            self.assertEqual(llm.list_models(self.cfg, "local1"), ["a", "b"])

    def test_model_switch_is_skipped_when_unchanged_or_switched_off(self):
        llm.LOADED_MODEL["name"] = "fake-model"
        with mock.patch("core.llm.subprocess.run") as run:
            llm.ensure_model_switched(self.cfg, "local1")
            run.assert_not_called()
        llm.LOADED_MODEL["name"] = None
        self.cfg["models"]["local1"]["unload_on_switch"] = False
        with mock.patch("core.llm.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            llm.ensure_model_switched(self.cfg, "local1")
            run.assert_not_called()
        self.assertEqual(llm.LOADED_MODEL["name"], "fake-model")

    def test_model_switch_unloads_once_when_the_model_changes(self):
        llm.LOADED_MODEL["name"] = "old-model"
        fake_run = mock.Mock(return_value=mock.Mock(returncode=0, stdout="", stderr=""))
        with mock.patch("core.llm.find_lms", return_value="/x/lms"), \
                mock.patch("core.llm.subprocess.run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            llm.ensure_model_switched(self.cfg, "local1")
            llm.ensure_model_switched(self.cfg, "local1")                     # same model now: no second unload
        self.assertEqual(fake_run.call_count, 1)
        self.assertEqual(fake_run.call_args[0][0], ["/x/lms", "unload", "--all"])


# ---------------------------------------------------------------- the character check

SMALL = {"mastered": "你好大", "learning": "我", "unknown": [{"char": "貓"}, {"char": "狗"}]}


class TestTextcheck(unittest.TestCase):
    def test_counts_split_into_mastered_learning_and_new(self):
        r = textcheck.check("你好貓貓。我狗鳥", SMALL, 10)
        self.assertEqual((r["total_chars"], r["unique"]), (7, 6))
        self.assertEqual(sorted(r["mastered"]), sorted("你好"))
        self.assertEqual(r["learning"], ["我"])
        self.assertEqual(sorted(r["unknown"]), sorted("貓狗鳥"))
        self.assertEqual(r["unknown"][-1], "鳥")                     # preferred-list characters first, the rest after
        self.assertEqual(sorted(r["in_pool"]), sorted("貓狗"))
        self.assertEqual(r["outside_pool"], ["鳥"])
        self.assertEqual({d["char"]: d["count"] for d in r["new_details"]}, {"貓": 2, "狗": 1, "鳥": 1})

    def test_target_is_a_count_of_different_new_characters(self):
        self.assertFalse(textcheck.check("貓狗鳥", SMALL, 2)["passed"])
        self.assertTrue(textcheck.check("貓狗鳥", SMALL, 3)["passed"])
        self.assertTrue(textcheck.check("貓貓貓貓", SMALL, 1)["passed"])       # repeats count once

    def test_punctuation_digits_and_spaces_are_not_restricted(self):
        r = textcheck.check("你好，123 !? 😀", SMALL, 0)
        self.assertEqual(r["unknown"], [])
        self.assertTrue(r["passed"])

    def test_notes_are_split_off_and_ignored(self):
        r = textcheck.check("你好\nNOTES: left out a minor detail 貓貓貓", SMALL, 0)
        self.assertEqual(r["notes"], "left out a minor detail 貓貓貓")
        self.assertEqual(r["unknown"], [])
        self.assertEqual(textcheck.check("你好\nNOTES: none", SMALL, 0)["notes"], "")
        self.assertEqual(textcheck.check("你好\nNOTES: None.", SMALL, 0)["notes"], "")

    def test_text_in_dnt_tags_is_not_counted_and_tags_are_removed(self):
        r = textcheck.check("<dnt>Elon Musk</dnt> 說你好", SMALL, 0, ["Elon Musk"])
        self.assertEqual(r["text"], "Elon Musk 說你好")
        self.assertEqual(r["dnt_found"], ["Elon Musk"])
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["warnings"], [])
        self.assertEqual(r["unknown"], ["說"])

    def test_missing_or_changed_dnt_text_is_a_problem(self):
        r = textcheck.check("<dnt>Elon Muskk</dnt> 你好", SMALL, 5, ["Elon Musk"])
        self.assertFalse(r["passed"])
        self.assertTrue(any("missing or changed" in p for p in r["problems"]))
        r = textcheck.check("伊隆 你好", SMALL, 5, ["Elon Musk"])
        self.assertTrue(any("Elon Musk" in p for p in r["problems"]))

    def test_chinese_wrapped_in_dnt_is_a_problem_and_extra_english_a_warning(self):
        r = textcheck.check("<dnt>貓</dnt> <dnt>NASA</dnt> 你好", SMALL, 5, [])
        self.assertTrue(any("only for English" in p for p in r["problems"]))
        self.assertTrue(any("extra <dnt> spans" in w for w in r["warnings"]))

    def test_dnt_spans_are_not_verified_when_the_story_was_not_tagged(self):
        r = textcheck.check("<dnt>Elon Musk</dnt> 你好", SMALL, 5, None)
        self.assertEqual(r["problems"], [])

    def test_unbalanced_tags_warn(self):
        r = textcheck.check("<dnt>Elon 你好", SMALL, 5)
        self.assertTrue(any("unbalanced" in w for w in r["warnings"]))

    def test_stray_english_outside_tags_warns(self):
        r = textcheck.check("你好 Starlink", SMALL, 5)
        self.assertTrue(any("English letters outside" in w for w in r["warnings"]))
        self.assertTrue(r["passed"])

    def test_simplified_only_characters_warn_or_fail_when_strict(self):
        r = textcheck.check("你好们", SMALL, 5)
        self.assertEqual(r["simplified"], ["们"])
        self.assertTrue(any("simplified-only" in w for w in r["warnings"]))
        self.assertTrue(r["passed"])
        r = textcheck.check("你好们", SMALL, 5, strict_simplified=True)
        self.assertFalse(r["passed"])
        self.assertTrue(any("simplified-only" in p for p in r["problems"]))

    def test_traditional_forms_are_not_flagged_as_simplified(self):
        self.assertEqual(textcheck.check("你們說這個，國家開會了。", SMALL, 99)["simplified"], [])

    def test_a_reply_with_no_chinese_fails(self):
        r = textcheck.check("only English here", SMALL, 5)
        self.assertFalse(r["passed"])
        self.assertTrue(any("no Chinese" in p for p in r["problems"]))

    def test_context_shows_the_clause_and_survives_dots_inside_protected_text(self):
        r = textcheck.check("你好。<dnt>U.S.</dnt> 沒有給貓警告，大", SMALL, 9, ["U.S."])
        d = {x["char"]: x for x in r["new_details"]}
        self.assertIn("U.S.", d["貓"]["context"])
        self.assertNotIn("<dnt>", d["貓"]["context"])

    def test_long_context_is_shortened_without_cutting_english_words(self):
        reply = "<dnt>Christopher Miller</dnt> of the <dnt>Financial Times</dnt> 說貓" + "你" * 30
        r = textcheck.check(reply, SMALL, 9, ["Christopher Miller", "Financial Times"])
        ctx = r["new_details"][0]["context"] if r["new_details"][0]["char"] == "貓" else r["new_details"][1]["context"]
        self.assertLessEqual(len(ctx), 44)

    def test_score_prefers_fewer_characters_outside_the_preferred_list(self):
        a = textcheck.check("貓狗", SMALL, 9)["score"]          # 0 outside, 2 new
        b = textcheck.check("鳥", SMALL, 9)["score"]            # 1 outside, 1 new
        self.assertLess(a, b)

    def test_report_names_the_new_characters_and_the_verdict(self):
        over = textcheck.format_report(textcheck.check("貓狗鳥", SMALL, 2))
        self.assertIn("OVER TARGET", over)
        self.assertIn("鳥  x1  (outside the list)", over)
        self.assertIn("貓  x1  (preferred list)", over)
        within = textcheck.format_report(textcheck.check("你好", SMALL, 2))
        self.assertIn("WITHIN TARGET", within)
        self.assertIn("  none", within)

    def test_target_and_tolerance_from_config(self):
        cfg = sample_cfg()
        self.assertEqual(textcheck.new_char_target(cfg), 50)
        self.assertEqual(textcheck.revert_tolerance(cfg), 10)
        del cfg["translation"]["revert_tolerance"]
        self.assertEqual(textcheck.revert_tolerance(cfg), 10)         # default
        cfg["translation"]["target_new_chars"] = "abc"
        with self.assertRaises(ConfigError):
            textcheck.new_char_target(cfg)
        cfg["translation"]["target_new_chars"] = -1
        with self.assertRaises(ConfigError):
            textcheck.new_char_target(cfg)
        cfg["translation"].pop("target_new_chars")
        with self.assertRaises(ConfigError):
            textcheck.new_char_target(cfg)

    def test_load_allowed_reports_a_missing_file_plainly(self):
        with self.assertRaises(RuntimeError) as cm:
            textcheck.load_allowed(path="/nonexistent/allowed_chars.json")
        self.assertIn("pleco_import", str(cm.exception))


class TestStarlinkArticle(unittest.TestCase):
    """The real translated article from the project, checked against the character lists in
    tests/fixtures/allowed_chars_sample.json (copied from translate_prompt.txt)."""

    @classmethod
    def setUpClass(cls):
        cls.allowed = json.loads(read_fixture("allowed_chars_sample.json"))
        cls.reply = read_fixture("starlink_reply.txt")
        cls.tagged_reply = read_fixture("starlink_reply_tagged.txt")
        cls.story_spans = textcheck.extract_dnt(read_fixture("starlink_tagged_story.txt"))

    def test_untagged_reply_counts(self):
        r = textcheck.check(self.reply, self.allowed, 50)
        self.assertGreater(r["total_chars"], 200)
        self.assertEqual(r["simplified"], [])
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["unique"], len(r["mastered"]) + len(r["learning"]) + len(r["unknown"]))
        self.assertEqual(r["passed"], len(r["unknown"]) <= 50)
        self.assertTrue(any("English letters outside" in w for w in r["warnings"]))      # names are untagged

    def test_tagged_reply_keeps_every_protected_span(self):
        self.assertEqual(len(self.story_spans), 27)
        r = textcheck.check(self.tagged_reply, self.allowed, 50, self.story_spans)
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["warnings"], [])
        self.assertEqual(sorted(r["dnt_found"]), sorted(self.story_spans))

    def test_tags_do_not_change_the_counts(self):
        a = textcheck.check(self.reply, self.allowed, 50)
        b = textcheck.check(self.tagged_reply, self.allowed, 50, self.story_spans)
        for key in ("total_chars", "unique", "mastered", "learning", "unknown", "percent"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["text"], self.reply.strip())


class TestRegression(unittest.TestCase):
    """New core/textcheck.py against the old check_text.py. Skipped unless the old file is found:
    set OLD_CHECK_TEXT=/path/check_text.py, or leave a copy at <project>/check_text.py or
    <project>/old/check_text.py."""

    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get("OLD_CHECK_TEXT"), BASE / "check_text.py", BASE / "old" / "check_text.py"]
        path = next((Path(c) for c in candidates if c and Path(c).is_file()), None)
        if path is None:
            raise unittest.SkipTest("old check_text.py not found (set OLD_CHECK_TEXT to run this)")
        spec = importlib.util.spec_from_file_location("old_check_text", path)
        cls.old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.old)
        cls.allowed = json.loads(read_fixture("allowed_chars_sample.json"))
        real = BASE / "data" / "allowed_chars.json"
        cls.real_allowed = json.loads(real.read_text(encoding="utf-8")) if real.is_file() else None

    FIELDS = ("body", "text", "notes", "dnt_found", "total_chars", "unique", "mastered", "learning",
              "unknown", "in_pool", "outside_pool", "new_details", "percent", "simplified", "problems",
              "warnings", "limit", "passed", "score")

    def compare(self, reply, allowed, max_new, expected=None, strict=False):
        old = self.old.check(reply, allowed, max_new, expected, strict)
        new = textcheck.check(reply, allowed, max_new, expected, strict)
        for key in self.FIELDS:
            self.assertEqual(old[key], new[key], f"field {key!r} differs")

    def test_untagged_article(self):
        for limit in (0, 10, 50, 200):
            self.compare(read_fixture("starlink_reply.txt"), self.allowed, limit)

    def test_tagged_article_with_span_check(self):
        spans = textcheck.extract_dnt(read_fixture("starlink_tagged_story.txt"))
        self.compare(read_fixture("starlink_reply_tagged.txt"), self.allowed, 50, spans)

    def test_strict_simplified(self):
        self.compare("你好们这说，貓", SMALL, 3, None, True)

    def test_with_the_real_character_lists_when_present(self):
        if not self.real_allowed:
            self.skipTest("data/allowed_chars.json not found")
        self.compare(read_fixture("starlink_reply.txt"), self.real_allowed, 50)


if __name__ == "__main__":
    unittest.main()
