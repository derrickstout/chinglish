"""Check the result of a REAL run of revert and judge_comprehension (step 7). Reads a job folder; no model calls.

    python3 run.py all --text samples/messy_newsletter.txt      # when the count is over target + tolerance,
                                                                # revert and judge_comprehension run too
    python3 tests/live_check_step7.py                           # newest job, or give a JOB_ID

PASS/FAIL  each reversion marked valid really is: <en> tags balanced, every <en> span is English from the
           original, <dnt> spans kept, fewer new characters than the winning translation (re-checked here)
           08_winner.txt matches a valid reversion (or, if none was valid, the winning translation)
LOOK       the new-character counts, the judge's reason, and the chosen text with English put back in [brackets]
Exit status is 0 when every PASS/FAIL line passes. Standard library only.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import job as jobmod                     # noqa: E402
from core import tags, textcheck                   # noqa: E402
from core.config import ConfigError, current_config  # noqa: E402
from stages import revert                          # noqa: E402

failed = 0


def report(ok, message):
    global failed
    failed += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + message)


def main(argv):
    try:
        cfg = current_config()
        job = jobmod.open_job(cfg, argv[1]) if len(argv) > 1 else jobmod.latest_job(cfg)
        allowed = textcheck.load_allowed(cfg)
        target = textcheck.new_char_target(cfg)
    except (ConfigError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
    if job is None:
        sys.exit("No jobs yet.")
    print(f"Job: {job.id}\n")
    if not job.exists("06_count.json"):
        report(False, "06_count.json exists (run: python3 run.py count --job " + job.id + ")")
        return 1
    counted = job.read_json("06_count.json")
    print(f"Winning translation: {counted['new_chars_count']} new characters "
          f"(target {target}, reversion above {counted['revert_threshold']}).")
    if not counted["revert_needed"]:
        print("No reversion was needed for this job, so there is nothing more to check. Use a longer article, "
              "or lower translation.target_new_chars, to see it run.")
        return 0
    if not job.exists("07_meta.json"):
        report(False, "07_meta.json exists (run: python3 run.py revert --job " + job.id + ")")
        return 1
    meta = job.read_json("07_meta.json")
    tagged = job.read("03_tagged.txt")
    english = textcheck.strip_dnt_tags(tagged)
    expected = textcheck.extract_dnt(tagged) if job.status().get("meta", {}).get("tagging") == "ok" else None
    print()
    for c in meta["candidates"]:
        label = f"reversion {c['n']} (temperature {c['temperature']:g})"
        if not c.get("file") or not job.exists(c["file"]):
            report(False, f"{label} exists; error: {c.get('error')}")
            continue
        problems, _, new = revert.verify(job.read(c["file"]).strip(), english, expected,
                                         counted["new_chars_count"], allowed, target)
        agree = (not problems) == bool(c["valid"])
        report(agree, f"{label}: {'valid' if not problems else 'rejected'}, {new} new characters, "
                      f"{c.get('en_spans', '?')} English insertions" + ("" if not problems else " - " + "; ".join(problems)))
    if not job.exists("08_winner.txt"):
        report(False, "08_winner.txt exists (run: python3 run.py judge_comprehension --job " + job.id + ")")
        return 1
    final = job.read("08_winner.txt").strip()
    info = job.read_json("08_judge.json")
    valid_texts = [job.read(c["file"]).strip() for c in meta["candidates"] if c.get("valid") and c.get("file")]
    expect = valid_texts if valid_texts else [job.read("05_winner.txt").strip()]
    report(final in expect, "08_winner.txt is " + ("one of the valid reversions" if valid_texts
                                                   else "the winning translation (no reversion was valid)"))
    print(f"\nLOOK  judge method: {info['method']}; reason: {info.get('reason') or '(none)'}\n")
    shown = textcheck.strip_dnt_tags(final).replace("<en>", "[").replace("</en>", "]")
    print("LOOK  final text (English put back is in [brackets]):\n      " + shown.replace("\n", "\n      ")[:1500])
    print(f"\nJob folder: {job.path}")
    print("Result: " + ("all checks passed" if not failed else f"{failed} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
