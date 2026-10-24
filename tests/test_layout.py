"""Step 0: confirm the starting folder is laid out as the build plan expects.

Run from the project folder (the one that holds core/, samples/, prompts/, data/):
    python3 -m unittest tests.test_layout -v
or, to run every test in tests/:
    python3 -m unittest discover tests

Standard library only. Nothing here changes any file.
"""
import json
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

KEPT_FILES = [
    "prompts/simplify.txt",
    "prompts/dnt.txt",
    "data/allowed_chars.json",
    "data/pleco_import.py",
]
REQUIRED_DIRS = ["core", "samples", "prompts", "data"]
ALLOWED_KEYS = ["mastered", "learning", "unknown"]


def _hint(rel):
    """Where a misplaced kept file might be, so the failure message says what to move."""
    name = Path(rel).name
    found = [str(p.relative_to(BASE)) for p in BASE.rglob(name)
             if p.is_file() and "tests" not in p.parts][:3]
    if found:
        return f" (found elsewhere: {', '.join(found)}; move it to {rel})"
    return ""


class TestLayout(unittest.TestCase):

    def test_required_folders_exist(self):
        for d in REQUIRED_DIRS:
            with self.subTest(folder=d):
                self.assertTrue((BASE / d).is_dir(), f"missing folder: {d}/")

    def test_kept_files_exist(self):
        for rel in KEPT_FILES:
            with self.subTest(file=rel):
                self.assertTrue((BASE / rel).is_file(), f"missing file: {rel}{_hint(rel)}")

    def test_kept_prompts_are_not_empty(self):
        for rel in ("prompts/simplify.txt", "prompts/dnt.txt"):
            p = BASE / rel
            if not p.is_file():
                continue                      # already reported by test_kept_files_exist
            with self.subTest(file=rel):
                self.assertTrue(p.read_text(encoding="utf-8").strip(), f"{rel} is empty")

    def test_allowed_chars_parses_and_has_lists(self):
        p = BASE / "data" / "allowed_chars.json"
        if not p.is_file():
            self.skipTest("data/allowed_chars.json is missing (see test_kept_files_exist)")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            self.fail(f"data/allowed_chars.json is not valid JSON: {e}")
        for key in ALLOWED_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, data, f"allowed_chars.json has no '{key}'")
        if "mastered" in data:
            self.assertTrue(data["mastered"], "'mastered' is empty")
        if "unknown" in data:
            self.assertIsInstance(data["unknown"], list, "'unknown' should be a list of {char, ...} entries")


if __name__ == "__main__":
    unittest.main()
