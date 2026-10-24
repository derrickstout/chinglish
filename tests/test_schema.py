"""Step A tests: core/schema.py (the rules for config.json), the shipped config.json, and `run.py config`.

Run from the project folder:
    python3 tests/test_schema.py -v
    python3 -m unittest discover tests
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import run                                         # noqa: E402
from core import schema                            # noqa: E402
from core.config import ConfigError                # noqa: E402
from tests.helpers import stage_cfg                # noqa: E402

SHIPPED = json.loads((BASE / "config.json").read_text(encoding="utf-8"))


def good():
    return copy.deepcopy(SHIPPED)


class TheShippedConfig(unittest.TestCase):
    def test_it_has_no_problems(self):
        self.assertEqual(schema.validate(SHIPPED), [])

    def test_it_defines_the_four_profiles(self):
        self.assertEqual(list(SHIPPED["models"]), ["local1", "local2", "cloud1", "cloud2"])

    def test_every_model_stage_names_a_defined_profile(self):
        for name in schema.STAGES:
            opts = schema.stage_opts(SHIPPED, name)
            if "model" in opts:
                self.assertIn(opts["model"], SHIPPED["models"])

    def test_the_helper_config_used_by_the_other_tests_is_valid_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(schema.validate(stage_cfg(tmp)), [])


class Profiles(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        p = schema.profile({"models": {"m": {"url": "http://x/v1", "model": "y"}}}, "m")
        self.assertEqual((p["temperature"], p["max_tokens"], p["server"], p["api_key_env"]), (0.3, 4096, "", ""))

    def test_url_and_model_are_required(self):
        cfg = {"models": {"m": {"url": "http://x/v1", "model": ""}}}
        with self.assertRaisesRegex(ConfigError, 'Set "models.m.model"'):
            schema.profile(cfg, "m")

    def test_an_undefined_profile_lists_the_defined_ones(self):
        with self.assertRaisesRegex(ConfigError, 'model "x" is not defined.*local1, local2, cloud1, cloud2'):
            schema.profile(SHIPPED, "x")

    def test_no_models_at_all_is_explained(self):
        with self.assertRaisesRegex(ConfigError, 'Set up "models"'):
            schema.profile({}, "local1")

    def test_any_number_of_profiles_with_any_names(self):
        cfg = good()
        for i in range(3, 9):
            cfg["models"][f"extra-{i}"] = {"url": "http://x/v1", "model": f"m{i}"}
        cfg["stages"]["tag"]["model"] = "extra-7"
        self.assertEqual(schema.validate(cfg), [])
        self.assertEqual(schema.profile(cfg, schema.stage_opts(cfg, "tag")["model"])["model"], "m7")

    def test_a_typo_in_a_profile_key_gets_a_suggestion(self):
        cfg = good()
        cfg["models"]["local1"]["max_token"] = 5
        self.assertTrue(any('"models.local1.max_token"' in p and 'did you mean "max_tokens"' in p
                            for p in schema.validate(cfg)))


class StageSettings(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        o = schema.stage_opts(SHIPPED, "translate")
        self.assertEqual((o["passes"], o["temperature"], o["pass_temp_step"], o["max_temperature"]), (3, 0.3, 0.3, 1.0))
        self.assertIsNone(o["max_tokens_override"])

    def test_the_judge_defaults(self):
        o = schema.stage_opts(SHIPPED, "judge_translation")
        self.assertEqual((o["temperature"], o["retries"]), (0.0, 1))

    def test_a_stage_with_no_model_setting_is_an_error(self):
        cfg = good()
        del cfg["stages"]["tag"]["model"]
        self.assertTrue(any('Set "stages.tag.model"' in p for p in schema.validate(cfg)))

    def test_a_stage_naming_an_undefined_profile_says_which_and_lists_the_defined_ones(self):
        cfg = good()
        cfg["stages"]["tag"]["model"] = "local9"
        problems = " ".join(schema.validate(cfg))
        self.assertIn('"stages.tag.model" is "local9", which is not defined in "models"', problems)
        self.assertIn("cloud2", problems)

    def test_an_unknown_stage_is_reported_with_a_suggestion(self):
        cfg = good()
        cfg["stages"]["translat"] = {"model": "local1"}
        self.assertTrue(any('"stages.translat" is not a stage' in p and 'did you mean "translate"' in p
                            for p in schema.validate(cfg)))

    def test_an_unknown_stage_setting_is_reported_with_a_suggestion(self):
        cfg = good()
        cfg["stages"]["judge_translation"]["retires"] = 2
        self.assertTrue(any('"stages.judge_translation.retires"' in p and 'did you mean "retries"' in p
                            for p in schema.validate(cfg)))

    def test_a_setting_that_does_not_belong_to_the_stage_is_reported(self):
        cfg = good()
        cfg["stages"]["tag"]["passes"] = 3
        self.assertTrue(any('"stages.tag.passes" is not a setting' in p for p in schema.validate(cfg)))

    def test_types_and_ranges(self):
        cases = [("translate", "passes", 0, "at least 1"), ("translate", "passes", 2.5, "whole number"),
                 ("translate", "passes", "3", "must be a number"), ("translate", "temperature", 3, "at most 2"),
                 ("translate", "temperature", True, "must be a number"), ("ingest", "max_chars", -1, "at least 0"),
                 ("judge_translation", "retries", 99, "at most"), ("tag", "max_tokens_override", 0, "at least 1")]
        for stage, key, value, expect in cases:
            with self.subTest(stage=stage, key=key, value=value):
                cfg = good()
                cfg["stages"][stage][key] = value
                self.assertTrue(any(f"stages.{stage}.{key}" in p and expect in p for p in schema.validate(cfg)),
                                schema.validate(cfg))

    def test_a_whole_number_written_as_a_float_is_accepted(self):
        cfg = good()
        cfg["stages"]["translate"]["passes"] = 3.0
        self.assertEqual(schema.stage_opts(cfg, "translate")["passes"], 3)

    def test_booleans_must_be_real_booleans(self):
        cfg = good()
        cfg["models"]["local1"]["unload_on_switch"] = "false"
        self.assertTrue(any("without quotes" in p for p in schema.validate(cfg)))

    def test_temperature_none_means_use_the_profiles(self):
        self.assertIsNone(schema.stage_opts(SHIPPED, "tag")["temperature"])

    def test_max_tokens_override_is_a_stage_setting_not_a_profile_setting(self):
        cfg = good()
        cfg["models"]["local1"]["max_tokens_override"] = 5
        self.assertTrue(any('"models.local1.max_tokens_override" is not a setting' in p for p in schema.validate(cfg)))


class PassTemperatures(unittest.TestCase):
    def opts(self, **kw):
        return dict({"passes": 3, "temperature": 0.3, "pass_temp_step": 0.3, "max_temperature": 1.0}, **kw)

    def test_the_default_is_the_old_three_temperatures(self):
        self.assertEqual(schema.pass_temperatures(self.opts()), [0.3, 0.6, 0.9])

    def test_no_floating_point_noise(self):
        self.assertEqual(schema.pass_temperatures(self.opts(passes=4, temperature=0.1, pass_temp_step=0.1)),
                         [0.1, 0.2, 0.3, 0.4])

    def test_the_cap(self):
        self.assertEqual(schema.pass_temperatures(self.opts(passes=4)), [0.3, 0.6, 0.9, 1.0])

    def test_one_pass_and_a_step_of_zero(self):
        self.assertEqual(schema.pass_temperatures(self.opts(passes=1)), [0.3])
        self.assertEqual(schema.pass_temperatures(self.opts(pass_temp_step=0)), [0.3, 0.3, 0.3])


class OldSettingsAreRecognised(unittest.TestCase):
    def test_old_sections_say_where_they_went(self):
        cfg = good()
        cfg["local_llm"] = {"url": "x"}
        cfg["cloud_llm"] = {"url": "y"}
        cfg["ingest"] = {"max_chars": 5}
        text = " ".join(schema.validate(cfg))
        self.assertIn('"local_llm" was replaced by a profile in "models"', text)
        self.assertIn('"cloud_llm" was replaced', text)
        self.assertIn('"ingest" moved to "stages.ingest"', text)

    def test_the_old_temperature_lists_say_where_they_went(self):
        cfg = good()
        cfg["translation"]["translate_temperatures"] = [0.3]
        cfg["translation"]["revert_temperatures"] = [0.3]
        text = " ".join(schema.validate(cfg))
        self.assertIn('"translation.translate_temperatures" was replaced by "stages.translate"', text)
        self.assertIn('"translation.revert_temperatures" was replaced by "stages.revert"', text)

    def test_an_unknown_section_gets_a_suggestion(self):
        cfg = good()
        cfg["stagez"] = {}
        self.assertTrue(any('"stagez" is not a section' in p and 'did you mean "stages"' in p for p in schema.validate(cfg)))

    def test_check_lists_every_problem_at_once(self):
        cfg = good()
        cfg["local_llm"] = {}
        cfg["stages"]["tag"]["model"] = "nope"
        with self.assertRaises(ConfigError) as cm:
            schema.check(cfg)
        text = str(cm.exception)
        self.assertIn("local_llm", text)
        self.assertIn('"nope"', text)
        self.assertGreaterEqual(text.count("\n  - "), 2)


class ConfigCommand(unittest.TestCase):
    def call(self, cfg):
        out = []
        code = run.main(["config"], cfg=cfg, out=out.append)
        return code, "\n".join(out)

    def test_a_good_config_lists_models_and_stages(self):
        code, text = self.call(good())
        self.assertEqual(code, 0)
        self.assertIn("local1", text)
        self.assertIn("used by simplify, tag, translate, revert", text)
        self.assertIn("(not used)", text)
        self.assertIn("config.json looks fine.", text)

    def test_it_shows_whether_each_key_is_set(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}):
            self.assertIn("key GEMINI_API_KEY: set", self.call(good())[1])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn("key GEMINI_API_KEY: NOT SET", self.call(good())[1])

    def test_problems_are_printed_and_the_exit_code_is_1(self):
        cfg = good()
        cfg["stages"]["tag"]["model"] = "nope"
        code, text = self.call(cfg)
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM:", text)
        self.assertIn("config.json has problems.", text)

    def test_every_other_command_refuses_to_start_with_a_bad_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = stage_cfg(tmp)
            cfg["stages"]["tag"]["model"] = "nope"
            out, err = [], io.StringIO()
            with contextlib.redirect_stderr(err):
                code = run.main(["ingest", "--latest"], cfg=cfg, out=out.append)
        self.assertEqual(code, 1)
        self.assertIn('"nope"', err.getvalue())


if __name__ == "__main__":
    unittest.main()
