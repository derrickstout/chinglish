"""Check the result of a REAL run of judge_translation (step 5). Reads a job folder; makes no model calls.

    export GEMINI_API_KEY="your key"                     # the profile named by stages.judge_translation.model must be set up in config.json
    python3 run.py translate --job ID
    python3 run.py judge_translation --job ID            # or: python3 run.py all ...
    python3 tests/live_check_step5.py                    # newest job, or give a JOB_ID

PASS/FAIL  05_winner.txt exists and matches one of the translations; the judge's answer was read (not the fallback)
LOOK       the judge's reason, and all candidates side by side with their new-character counts
Exit status is 0 when every PASS/FAIL line passes. Standard library only.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core import job as jobmod                     # noqa: E402
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
    except ConfigError as e:
        sys.exit(f"Error: {e}")
    if job is None:
        sys.exit("No jobs yet.")
    print(f"Job: {job.id}\n")
    if not (job.exists("05_winner.txt") and job.exists("05_judge.json")):
        report(False, "05_winner.txt exists (run: python3 run.py judge_translation --job " + job.id + ")")
        return 1
    info, meta = job.read_json("05_judge.json"), job.read_json("04_meta.json")
    winner = job.read("05_winner.txt").strip()
    cands = {c["n"]: c for c in meta["candidates"] if c.get("file") and job.exists(c["file"])}
    report(info["winner"] in cands and job.read(cands[info["winner"]]["file"]).strip() == winner,
           f"05_winner.txt is translation {info['winner']} word for word")
    report(info["method"] in ("judge", "only"),
           f"judge method: {info['method']}" + (" (the reply could not be read; see 05_judge.json)"
                                                if info["method"] == "fallback" else ""))
    print(f"\nLOOK  judge's reason: {info.get('reason') or '(none)'}\n")
    for n, c in cands.items():
        mark = "  <-- WINNER" if n == info["winner"] else ""
        print(f"LOOK  translation {n}, t={c['temperature']:g}, {c['new_chars']} new characters{mark}")
        print("      " + job.read(c["file"]).strip().replace("\n", "\n      ")[:500] + "\n")
    print("Result: " + ("all checks passed" if not failed else f"{failed} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
