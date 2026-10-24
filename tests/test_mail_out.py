"""Step 9 tests: stages/mail_out.py. No mail server is contacted: smtplib is replaced by a recording fake.

Run from the project folder:
    python3 tests/test_mail_out.py -v
    python3 -m unittest discover tests
"""
import email
import os
import smtplib
import sys
import tempfile
import unittest
from email import policy
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pipeline                                    # noqa: E402
from core.config import ConfigError               # noqa: E402
from stages import mail_out                        # noqa: E402
from tests.helpers import job_with, stage_cfg      # noqa: E402

RESULT = {
    "translation": "Dana去了市場。\n\n她買了蘋果。",
    "english": "Dana went to the market.\n\nShe bought apples.",
    "original": "Dana walked to the market <b>& bought</b> apples.",
    "notes": ["WARNINGS:", "  - only the beginning was translated", "", "Translation 2 of 3 (temperature 0.6)."],
}
RESULT_TXT = "TRANSLATION\n\nDana去了市場。\n\nNOTES\n\nfine\n"


class FakeSMTP:
    """Stands in for smtplib.SMTP and SMTP_SSL, and records what it is asked to do."""
    instances = []
    login_error = None
    connect_error = None

    def __init__(self, host, port, timeout=None, context=None):
        if FakeSMTP.connect_error:
            raise FakeSMTP.connect_error
        self.host, self.port, self.context = host, port, context
        self.calls, self.sent = [], []
        FakeSMTP.instances.append(self)

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        if FakeSMTP.login_error:
            raise FakeSMTP.login_error
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.calls.append("quit")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = stage_cfg(self.tmp.name)
        self.cfg["mail"] = {"username": "me@example.com", "to": "me@example.com"}
        FakeSMTP.instances, FakeSMTP.login_error, FakeSMTP.connect_error = [], None, None
        for name in ("SMTP", "SMTP_SSL"):
            p = mock.patch.object(smtplib, name, FakeSMTP)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {"MAIL_PASSWORD": "app-pass"})
        env.start()
        self.addCleanup(env.stop)
        self.job = job_with(self.cfg, **{"09_result.txt": RESULT_TXT})
        self.job.write_json("09_result.json", RESULT)

    def enable(self, **extra):
        self.cfg["stages"]["mail_out"] = {"enabled": True}
        self.cfg["mail"].update(**extra)


class Message(Base):
    def msg(self):
        return mail_out.build_message(self.cfg, self.job)

    def test_addresses_and_default_subject(self):
        m = self.msg()
        self.assertEqual((m["From"], m["To"]), ("me@example.com", "me@example.com"))
        self.assertEqual(m["Subject"], "Chinese study: Dana walked to the market <b>& bought</b> apples.")

    def test_a_job_can_name_its_own_recipient_and_subject_and_thread(self):
        self.job.set_meta(reply_to="friend@example.org", email_subject="Starlink article", message_id="<abc@x>")
        m = self.msg()
        self.assertEqual((m["To"], m["Subject"]), ("friend@example.org", "Re: Starlink article"))
        self.assertEqual((m["In-Reply-To"], m["References"]), ("<abc@x>", "<abc@x>"))

    def test_an_existing_re_is_not_doubled(self):
        self.job.set_meta(email_subject="RE: hello")
        self.assertEqual(self.msg()["Subject"], "RE: hello")

    def test_the_plain_text_part_is_the_result_file_and_there_is_an_html_part(self):
        m = self.msg()
        self.assertEqual(m.get_body(("plain",)).get_content().strip(), RESULT_TXT.strip())
        self.assertIn("Dana去了市場。", m.get_body(("html",)).get_content())

    def test_the_html_escapes_text_and_shows_warnings_apart_from_notes(self):
        page = mail_out.body_html(RESULT)
        self.assertIn("&lt;b&gt;&amp; bought&lt;/b&gt;", page)
        self.assertNotIn("<b>& bought", page)
        self.assertIn("Check this:", page)
        self.assertIn("<li>only the beginning was translated</li>", page)
        self.assertIn("<li>Translation 2 of 3 (temperature 0.6).</li>", page)
        self.assertLess(page.index("Translation"), page.index("English (simplified)"))
        self.assertLess(page.index("English (simplified)"), page.index("Original article"))

    def test_no_warning_box_when_there_are_none(self):
        page = mail_out.body_html(dict(RESULT, notes=["fine"]))
        self.assertNotIn("Check this:", page)

    def test_chinese_survives_being_written_and_read_back(self):
        raw = self.msg().as_bytes()
        back = email.message_from_bytes(raw, policy=policy.default)
        self.assertIn("她買了蘋果。", back.get_body(("html",)).get_content())

    def test_a_missing_recipient_or_sender_is_a_plain_error(self):
        self.cfg["mail"].update(to="")
        with self.assertRaisesRegex(ConfigError, "No recipient"):
            self.msg()
        self.cfg["mail"].update(to="me@example.com", username="", **{"from": ""})
        with self.assertRaisesRegex(ConfigError, "No sender"):
            self.msg()

    def test_from_overrides_username(self):
        self.cfg["mail"]["from"] = "bot@example.com"
        self.assertEqual(self.msg()["From"], "bot@example.com")


class Stage(Base):
    def test_a_dry_run_saves_the_email_but_does_not_send_it(self):
        self.enable()
        self.cfg["stages"]["mail_out"]["dry_run"] = True
        note = mail_out.run(self.cfg, self.job)
        self.assertTrue(self.job.exists("10_email.eml"))
        info = self.job.read_json("10_sent.json")
        self.assertEqual((info["sent"], info["dry_run"]), (False, True))
        self.assertIn("dry run", note)
        self.assertEqual(FakeSMTP.instances, [])

    def test_with_mail_on_and_no_settings_it_is_an_error(self):
        self.enable()
        self.cfg["mail"] = {}
        with self.assertRaises(ConfigError):
            mail_out.run(self.cfg, self.job)

    def test_a_real_send_logs_in_and_sends_once(self):
        self.enable()
        note = mail_out.run(self.cfg, self.job)
        smtp = FakeSMTP.instances[0]
        self.assertEqual((smtp.host, smtp.port), ("smtp.office365.com", 587))
        self.assertEqual(smtp.calls, ["ehlo", "starttls", "ehlo", ("login", "me@example.com", "app-pass"), "quit"])
        self.assertEqual(len(smtp.sent), 1)
        self.assertEqual(smtp.sent[0]["To"], "me@example.com")
        info = self.job.read_json("10_sent.json")
        self.assertTrue(info["sent"] and info["time"])
        self.assertTrue(note.startswith("sent to me@example.com"))

    def test_ssl_and_custom_server_settings(self):
        self.enable(smtp_host="smtp.example.com", smtp_port=465, security="ssl")
        mail_out.run(self.cfg, self.job)
        smtp = FakeSMTP.instances[0]
        self.assertEqual((smtp.host, smtp.port), ("smtp.example.com", 465))
        self.assertNotIn("starttls", smtp.calls)

    def test_security_none_skips_starttls(self):
        self.enable(security="none")
        mail_out.run(self.cfg, self.job)
        self.assertNotIn("starttls", FakeSMTP.instances[0].calls)

    def test_the_password_variable_name_comes_from_config(self):
        self.enable(password_env="OTHER_PASS")
        with mock.patch.dict(os.environ, {"OTHER_PASS": "p2"}):
            mail_out.run(self.cfg, self.job)
        self.assertIn(("login", "me@example.com", "p2"), FakeSMTP.instances[0].calls)

    def test_a_missing_password_says_how_to_set_it_and_connects_to_nothing(self):
        self.enable()
        with mock.patch.dict(os.environ, {"MAIL_PASSWORD": ""}):
            with self.assertRaisesRegex(ConfigError, 'export MAIL_PASSWORD="your app password"'):
                mail_out.run(self.cfg, self.job)
        self.assertEqual(FakeSMTP.instances, [])
        self.assertFalse(self.job.exists("10_sent.json"))

    def test_a_bad_security_setting_is_an_error(self):
        self.enable(security="tls13")
        with self.assertRaisesRegex(ConfigError, "mail.security"):
            mail_out.run(self.cfg, self.job)

    def test_a_refused_login_explains_app_passwords(self):
        self.enable()
        FakeSMTP.login_error = smtplib.SMTPAuthenticationError(535, b"bad credentials")
        with self.assertRaisesRegex(RuntimeError, "refused the login.*app password"):
            mail_out.run(self.cfg, self.job)
        self.assertFalse(self.job.exists("10_sent.json"))

    def test_an_unreachable_server_is_a_plain_error(self):
        self.enable()
        FakeSMTP.connect_error = OSError("connection refused")
        with self.assertRaisesRegex(RuntimeError, "Could not send through smtp.office365.com:587"):
            mail_out.run(self.cfg, self.job)

    def test_a_missing_result_says_what_to_run(self):
        job = job_with(self.cfg)
        with self.assertRaises(FileNotFoundError):
            mail_out.run(self.cfg, job)


class ThroughThePipeline(Base):
    def test_mail_out_is_the_last_stage_and_needs_the_package_output(self):
        stage = pipeline.get_stage("mail_out")
        self.assertEqual(pipeline.names()[-1], "mail_out")
        self.assertEqual(stage.needs, ("09_result.json",))

    def test_it_runs_after_package_and_is_marked_done(self):
        self.enable()
        self.cfg["stages"]["mail_out"]["dry_run"] = True
        self.assertEqual(pipeline.run_stage(self.cfg, self.job, "mail_out", out=lambda s: None), "done")
        self.assertTrue(self.job.stage_done("mail_out"))

    def test_while_the_switch_is_off_nothing_is_built_or_sent(self):
        out = []
        result = pipeline.run_stage(self.cfg, self.job, "mail_out", out=out.append)
        self.assertEqual(result, "turned_off")
        self.assertFalse(self.job.exists("10_email.eml"))
        self.assertEqual(FakeSMTP.instances, [])
        self.assertEqual(self.job.status()["stages"]["mail_out"]["status"], "skipped")
        self.assertIn("turned off", out[0])

    def test_it_cannot_run_before_package(self):
        self.enable()
        job = job_with(self.cfg)
        with self.assertRaisesRegex(pipeline.StageError, r"mail_out needs 09_result.json.*run.py package --job"):
            pipeline.run_stage(self.cfg, job, "mail_out", out=lambda s: None)


class TestMessageCommand(Base):
    def test_the_test_message_goes_through_the_same_sender(self):
        with mock.patch("core.config.current_config", return_value=self.cfg):
            mail_out.main(["--test"])
        smtp = FakeSMTP.instances[0]
        self.assertEqual(smtp.sent[0]["Subject"], "Chinese Study Buddy: test message")
        self.assertIn("login", [c[0] for c in smtp.calls if isinstance(c, tuple)])

    def test_a_missing_password_is_a_clean_exit_not_a_traceback(self):
        with mock.patch("core.config.current_config", return_value=self.cfg), \
                mock.patch.dict(os.environ, {"MAIL_PASSWORD": ""}):
            with self.assertRaises(SystemExit) as cm:
                mail_out.main(["--test"])
        self.assertIn("mail password is not set", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
