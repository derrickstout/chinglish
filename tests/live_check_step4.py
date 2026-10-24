"""Check the result of a REAL run of translate (step 4). Not a unit test: it reads a job folder.

    python3 run.py all --text samples/messy_newsletter.txt     # stops at "judge_translation not built yet"
    python3 tests/live_check_step4.py                          # checks the newest job
    python3 tests/live_check_step4.py JOB_ID                   # or a named one

PASS/FAIL  04_meta.json and at least one translation exist; every translation has Chinese, no simplified-only
           characters, no leftover English outside <dnt> tags, and keeps every protected span
LOOK       new-character count per translation against the target, the new characters, and each text
Exit status is 0 when every PASS/FAIL line passes. Standard library only.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import job as jobmod                     # noqa: E402
from core import textcheck                         # noqa: E402
from core.config import ConfigError, current_config  # noqa: E402

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
        sys.exit("No jobs yet. Run: python3 run.py all --text samples/messy_newsletter.txt")
    print(f"Job: {job.id}\n")
    if not job.exists("04_meta.json"):
        report(False, "04_meta.json exists (run: python3 run.py translate --job " + job.id + ")")
        return 1
    meta = job.read_json("04_meta.json")
    tagged = job.read("03_tagged.txt")
    expected = textcheck.extract_dnt(tagged) if meta.get("tagging") == "ok" else None
    print(f"Target: {target} new characters (soft). Prompt about {meta.get('prompt_tokens', '?'):,} tokens.\n")
    made = 0
    for c in meta["candidates"]:
        label = f"translation {c['n']} (temperature {c['temperature']:g}, {c.get('seconds', '?')}s)"
        if not c.get("file") or not job.exists(c["file"]):
            report(False, f"{label} exists; error: {c.get('error')}")
            continue
        made += 1
        r = textcheck.check(job.read(c["file"]), allowed, target, expected_dnt=expected, strict_simplified=True)
        report(not r["problems"], f"{label}: no problems" + ("" if not r["problems"] else " - " + "; ".join(r["problems"])))
        for w in r["warnings"]:
            print("      warning: " + w)
        print(f"LOOK  {r['total_chars']} characters, {len(r['unknown'])} new "
              f"({len(r['outside_pool'])} outside your preferred list): {''.join(r['unknown']) or '(none)'}")
        print("      " + r["text"].replace("\n", "\n      ")[:600] + "\n")
    report(made > 0, f"at least one translation was made ({made} of {len(meta['candidates'])})")
    print(f"Job folder: {job.path}")
    print("Result: " + ("all checks passed" if not failed else f"{failed} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
