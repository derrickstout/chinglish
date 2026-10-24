"""Check the result of a REAL run of simplify and tag (step 3). Not a unit test: it reads a job folder.

    python3 run.py all --text samples/messy_newsletter.txt     # or: ingest, simplify, tag one at a time
    python3 tests/live_check_step3.py                          # checks the newest job
    python3 tests/live_check_step3.py JOB_ID                   # or a named one

It checks what a script can check, and prints what only you can judge:
  PASS/FAIL  02_simplified.txt exists, has text, no markdown, no Chinese, no chat-style introduction
  PASS/FAIL  03_tagged.txt has balanced <dnt> tags and equals 02_simplified.txt once they are removed
  LOOK       every protected span, and capitalised words left outside the tags (names that may have been missed)
Exit status is 0 when every PASS/FAIL line passes. Standard library only.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import job as jobmod                     # noqa: E402
from core import textcheck                         # noqa: E402
from core.config import ConfigError, current_config  # noqa: E402
from stages import simplify                        # noqa: E402

COMMON = {"The", "A", "An", "He", "She", "It", "They", "We", "I", "In", "On", "At", "This", "That", "These",
          "Those", "But", "And", "Also", "Like", "One", "Two", "Some", "Many", "After", "Before", "When",
          "While", "If", "As", "For", "From", "With", "There", "Monday", "Tuesday", "Wednesday", "Thursday",
          "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December", "President", "Mayor", "Mr", "Mrs", "Ms", "Dr"}

failed = 0


def report(ok, message):
    global failed
    failed += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + message)


def main(argv):
    try:
        cfg = current_config()
        job = jobmod.open_job(cfg, argv[1]) if len(argv) > 1 else jobmod.latest_job(cfg)
    except ConfigError as e:
        sys.exit(f"Error: {e}")
    if job is None:
        sys.exit("No jobs yet. Run: python3 run.py all --text samples/messy_newsletter.txt")
    print(f"Job: {job.id}\n")

    # ---- simplify
    if not job.exists("02_simplified.txt"):
        report(False, "02_simplified.txt exists (run: python3 run.py simplify --job " + job.id + ")")
        return 1
    simple = job.read("02_simplified.txt").strip()
    clean = job.read("01_clean.txt") if job.exists("01_clean.txt") else ""
    report(bool(simple), f"02_simplified.txt has text ({len(simple.split()):,} words; 01_clean.txt has "
                         f"{len(clean.split()):,})")
    markdown = re.findall(r"\*\*|__|(?m:^#{1,6} )|`", simple)
    report(not markdown, "no markdown marks (** __ # `) in 02_simplified.txt")
    report(not re.search(r"[一-鿿]", simple), "no Chinese characters in 02_simplified.txt")
    for w in simplify.check_reply(clean, simple):
        report(False, "simplify warning: " + w)

    # ---- tag
    if not job.exists("03_tagged.txt"):
        report(False, "03_tagged.txt exists (run: python3 run.py tag --job " + job.id + ")")
        return 1
    tagged = job.read("03_tagged.txt").strip()
    meta = job.status().get("meta", {})
    opens, closes = tagged.count("<dnt>"), tagged.count("</dnt>")
    pairs = len(textcheck.DNT_SPAN.findall(tagged))
    report(opens == closes == pairs, f"<dnt> tags are balanced ({opens} opening, {closes} closing)")
    same = " ".join(textcheck.strip_dnt_tags(tagged).split()) == " ".join(simple.split())
    report(same, "03_tagged.txt equals 02_simplified.txt once the tags are removed")
    if meta.get("tagging") == "discarded":
        report(False, f"tagging was discarded by the stage ({meta.get('tagging_problem')}); "
                      "see 03_tag_reply_raw.txt in the job folder")

    # ---- for you to judge
    spans = textcheck.extract_dnt(tagged)
    print(f"\nLOOK  protected spans ({len(spans)}):")
    for s in spans:
        print("        " + s)
    outside = textcheck.DNT_SPAN.sub(" ", tagged)
    words = []
    for chunk in re.split(r"(?<=[.!?\"\u201d])\s+|\n+", outside):      # the first word of a sentence is skipped
        for w in chunk.split()[1:]:
            w = w.strip(",;:()\"\u201c\u201d")
            if re.match(r"[A-Z][A-Za-z.'\u2019-]+$", w):
                words.append(w)
    missed = sorted({w for w in words if w.rstrip(".") not in COMMON})
    print("\nLOOK  capitalised words NOT inside tags (names or acronyms that may have been missed):")
    print("        " + (", ".join(missed[:40]) if missed else "(none)"))
    print(f"\nJob folder: {job.path}")
    print("Result: " + ("all checks passed" if not failed else f"{failed} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
