"""Step 2 tests: stages/ingest.py (cleaning raw newsletter text into plain text).

Run from the project folder:
    python3 tests/test_ingest.py -v
    python3 -m unittest discover tests

No model or network is used. If samples/messy_newsletter.txt exists, one extra test runs ingest on
it and checks general properties (no URLs, no unsubscribe or view-in-browser lines, no tags).
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import job as jobmod                     # noqa: E402
from core.config import ConfigError                # noqa: E402
from stages import ingest                          # noqa: E402

FIX = BASE / "tests" / "fixtures"

PARA_1 = ("The city council voted on Tuesday to expand the downtown bike lane network after months of "
          "public debate. The plan adds twelve miles of protected lanes and is expected to cost eighteen "
          "million dollars over three years.")
PARA_2 = ("Supporters say the lanes will make streets safer. Opponents worry about lost parking for small "
          "shops. \"We will watch the numbers closely,\" said Mayor Dana Whitfield")

UNWANTED = ["unsubscribe", "view this email", "view in browser", "http", "www.", "[1]", "[image", "©",
            "all rights reserved", "mailing address", "preferences", "advertisement", "display:none",
            "do not show this", "Preview:", "<", ">"]


def read(name):
    return (FIX / name).read_text(encoding="utf-8")


class TestPlainText(unittest.TestCase):
    def setUp(self):
        self.text, self.report = ingest.clean(read("messy_synthetic.txt"))

    def test_exact_output(self):
        expected = "\n\n".join([
            "THE WEEKLY DISPATCH\nIssue 42 | October 16, 2026",
            "Hi friend,",
            PARA_1,
            PARA_2 + ".",
            "The privacy policy for the new bike-share app was also updated on Tuesday.",
        ])
        self.assertEqual(self.text, expected)

    def test_nothing_unwanted_is_left(self):
        for bad in UNWANTED:
            self.assertNotIn(bad.lower(), self.text.lower(), f"{bad!r} is still in the cleaned text")

    def test_hard_wrapped_paragraphs_are_joined(self):
        self.assertIn(PARA_1, self.text.split("\n\n"))

    def test_a_sentence_about_privacy_is_kept(self):
        self.assertIn("The privacy policy for the new bike-share app", self.text)

    def test_report_says_what_happened(self):
        reasons = {r for r, _ in self.report.removed}
        self.assertTrue({"view in browser", "ad label", "social links", "link label"} <= reasons)
        self.assertIn("Copyright", self.report.footer_start)
        self.assertGreater(self.report.footer_lines, 3)
        self.assertEqual(self.report.before, len(read("messy_synthetic.txt")))
        self.assertEqual(self.report.after, len(self.text))

    def test_cleaning_twice_changes_nothing(self):
        again, _ = ingest.clean(self.text)
        self.assertEqual(again, self.text)


class TestHtml(unittest.TestCase):
    def setUp(self):
        self.text, self.report = ingest.clean(read("messy_synthetic.html"))

    def test_exact_output(self):
        expected = "\n\n".join([
            "THE WEEKLY DISPATCH",
            "Issue 42 | October 16, 2026",
            "Hi friend,",
            PARA_1,
            PARA_2 + " & the rest of the council.",
            "Read the full story on the council’s site for more.",
        ])
        self.assertEqual(self.text, expected)

    def test_nothing_unwanted_is_left(self):
        for bad in UNWANTED:
            self.assertNotIn(bad.lower(), self.text.lower(), f"{bad!r} is still in the cleaned text")

    def test_detected_as_html_and_not_unwrapped_as_plain_text(self):
        self.assertTrue(self.report.was_html)
        self.assertGreaterEqual(len(self.text.split("\n\n")), 5)

    def test_entities_and_nbsp_become_text(self):
        self.assertIn("three years", self.text)
        self.assertIn("& the rest", self.text)

    def test_plain_text_with_one_angle_bracket_is_not_html(self):
        self.assertFalse(ingest.looks_like_html("If x < 5 and y > 3 then <b>maybe</b>."))
        self.assertFalse(ingest.looks_like_html("See <https://example.com/a> for more."))
        self.assertTrue(ingest.looks_like_html("<p>One</p><p>Two</p>"))


class TestRules(unittest.TestCase):
    def clean(self, text, **kw):
        return ingest.clean(text, **kw)[0]

    def test_urls_in_every_form(self):
        text = ("Read [the report](https://example.com/r) now.\n"
                "Also see https://example.com/a?b=1 and www.example.org today.\n"
                "Or <https://example.com/angle> here.")
        out = self.clean(text)
        self.assertIn("Read the report now.", out)
        self.assertNotIn("http", out)
        self.assertNotIn("www", out)
        self.assertIn("Also see and today.", out)

    def test_invisible_characters_and_odd_spaces(self):
        out = self.clean("Hello​ world­!\r\n͏ ͏\r\nSecond line")
        self.assertEqual(out, "Hello world!\n\nSecond line")

    def test_footer_after_early_marker_is_not_cut(self):
        # "Unsubscribe" at the top: only that line goes, the article stays.
        article = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
        out = self.clean("Not interested? Unsubscribe\n\n" + article)
        self.assertNotIn("Unsubscribe", out)
        self.assertEqual(out.count("Paragraph"), 6)

    def test_late_marker_followed_by_real_prose_cuts_only_the_line(self):
        article = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
        long_tail = "Tail " + "sentence about something real. " * 30
        out = self.clean(article + "\n\nUnsubscribe here\n\n" + long_tail)
        self.assertNotIn("Unsubscribe", out)
        self.assertIn("Tail sentence about something real.", out)

    def test_late_marker_followed_by_footer_cuts_everything_after(self):
        article = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
        out = self.clean(article + "\n\nAll rights reserved.\n\n123 Main St\nSpringfield\n\nUnsubscribe")
        self.assertNotIn("Main St", out)
        self.assertEqual(out.count("Paragraph"), 6)

    def test_social_rows_go_but_headlines_stay(self):
        out = self.clean("Facebook | Twitter | Instagram\n\nFacebook fined $1 billion\n\nFollow us on Instagram\n\n"
                         "Share this story on Twitter\n\nTwitter was down for an hour.")
        self.assertEqual(out, "Facebook fined $1 billion\n\nTwitter was down for an hour.")

    def test_boilerplate_words_inside_a_long_sentence_are_kept(self):
        sentence = ("Critics noted that the app makes it hard to unsubscribe from its marketing messages, "
                    "which the regulator said breaks the rules that apply to every company in the sector today, "
                    "and which it plans to review again with industry groups before the end of next year.")
        self.assertGreater(len(sentence), ingest.BOILERPLATE_MAX_LINE)
        self.assertEqual(self.clean("Intro line.\n\n" + sentence), "Intro line.\n\n" + sentence)

    def test_list_items_and_short_lines_are_not_unwrapped(self):
        text = "- first item that is long enough to look like a wrapped line of text here\n- second item"
        self.assertEqual(self.clean(text), text)
        short = "Line one\nLine two\nLine three"
        self.assertEqual(self.clean(short), short)

    def test_wrapped_lines_that_end_in_periods_are_not_joined(self):
        text = ("This line is long enough to pass the width test for sure.\n"
                "This one is also long enough to pass the width test too.\n"
                "And the last one ends here.")
        self.assertEqual(self.clean(text), text)

    def test_separator_lines_and_reference_markers(self):
        out = self.clean("Intro [12] text.\n\n----\n\n* * *\n\n[3] https://x.example/ref\n\nOutro.")
        self.assertEqual(out, "Intro text.\n\nOutro.")

    def test_chinese_lines_count_as_content(self):
        self.assertEqual(self.clean("你好。\n\n---\n\n再見。"), "你好。\n\n再見。")


class TestLengthCap(unittest.TestCase):
    PARAS = [f"Paragraph {i}. " + "x" * 80 for i in range(10)]

    def test_cut_at_a_paragraph_break(self):
        text = "\n\n".join(self.PARAS)
        out, report = ingest.clean(text, max_chars=400)
        self.assertLessEqual(len(out), 400)
        self.assertTrue(out.endswith("x"))
        self.assertEqual(out.count("Paragraph"), len(out.split("\n\n")))
        self.assertEqual(report.truncated_to, len(out))
        self.assertIn("CUT TO THE FIRST", report.summary())

    def test_zero_turns_the_cap_off(self):
        text = "\n\n".join(self.PARAS)
        out, report = ingest.clean(text, max_chars=0)
        self.assertEqual(out, text)
        self.assertEqual(report.truncated_to, 0)

    def test_one_huge_paragraph_is_cut_at_a_sentence(self):
        text = "Short opening sentence here. " * 50
        out, _ = ingest.clean(text, max_chars=300)
        self.assertLessEqual(len(out), 300)
        self.assertTrue(out.endswith("."))


class TestStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {"paths": {"jobs": self.tmp.name}}

    def job(self, text):
        return jobmod.new_job(self.cfg, text)

    def test_run_writes_the_clean_file_and_returns_a_note(self):
        job = self.job(read("messy_synthetic.txt"))
        note = ingest.run(self.cfg, job)
        self.assertEqual(job.read("01_clean.txt"), ingest.clean(read("messy_synthetic.txt"))[0] + "\n")
        self.assertIn("1,122 -> 522 characters", note)
        log = job.read("log.txt")
        self.assertIn("ingest removed [view in browser]", log)
        self.assertIn("ingest cut footer", log)

    def test_source_is_never_changed(self):
        job = self.job(read("messy_synthetic.txt"))
        ingest.run(self.cfg, job)
        self.assertEqual(job.read("00_source.txt"), read("messy_synthetic.txt"))

    def test_empty_source_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "empty"):
            ingest.run(self.cfg, self.job("   \n  "))

    def test_only_boilerplate_is_an_error_that_points_at_the_tool(self):
        with self.assertRaisesRegex(RuntimeError, "python3 -m stages.ingest"):
            ingest.run(self.cfg, self.job("View in browser\n\nUnsubscribe\n\n© 2026 Acme. All rights reserved."))

    def test_max_chars_comes_from_config(self):
        job = self.job("\n\n".join(TestLengthCap.PARAS))
        cfg = dict(self.cfg, stages={"ingest": {"max_chars": 300}})
        note = ingest.run(cfg, job)
        self.assertIn("CUT TO THE FIRST", note)
        self.assertLessEqual(len(job.read("01_clean.txt").strip()), 300)

    def test_bad_max_chars_is_a_clear_error(self):
        for bad in ("lots", -5):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ConfigError, "stages.ingest.max_chars"):
                    ingest.run(dict(self.cfg, stages={"ingest": {"max_chars": bad}}), self.job("Some text."))

    def test_warns_when_most_of_the_text_was_removed(self):
        junk = "\n".join(["Unsubscribe"] * 5 + ["Share on Facebook | Twitter"] * 20) + "\n\nOne real sentence stays."
        _, report = ingest.clean("Intro line here.\n\n" + junk)
        self.assertIn("WARNING", report.summary())


class TestRealSample(unittest.TestCase):
    """Runs on samples/messy_newsletter.txt when it exists. Checks general properties only."""

    def test_properties(self):
        path = BASE / "samples" / "messy_newsletter.txt"
        if not path.is_file():
            self.skipTest("samples/messy_newsletter.txt not found")
        source = path.read_text(encoding="utf-8-sig")
        text, report = ingest.clean(source, max_chars=0)
        self.assertTrue(text.strip(), "nothing was left of the sample")
        self.assertNotRegex(text, r"https?://|www\.", "a URL survived")
        self.assertNotRegex(text, r"</?[a-zA-Z][^>]*>", "an HTML tag survived")
        self.assertNotRegex(text, r"\n{3,}", "three blank lines in a row")
        for line in text.split("\n"):
            low = line.lower()
            self.assertNotIn("unsubscribe", low, f"boilerplate line survived: {line[:100]}")
            self.assertNotRegex(low, r"view (?:this )?(?:e-?mail )?in (?:your )?browser", f"survived: {line[:100]}")
        self.assertLess(len(text), len(source), "nothing was removed from the messy sample")
        print(f"\n  real sample: {report.summary()}")


if __name__ == "__main__":
    unittest.main()
