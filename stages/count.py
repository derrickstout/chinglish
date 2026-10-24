"""Stage 6, count: count the new characters in the winning translation and decide whether English
reversion is needed. No model is used.

    python3 run.py count --job ID           # reads 05_winner.txt, writes 06_count.json and 06_count.txt

A "new" character is a Chinese character outside your Mastered and Learning lists in allowed_chars.json.
Reversion is needed when the new-character count is more than translation.target_new_chars plus
translation.revert_tolerance (config.json; defaults 50 and 10). pipeline.py reads "revert_needed" in
06_count.json to decide whether the revert and judge_comprehension stages run.

06_count.json holds the numbers, every new character with how often it occurs and a snippet of context
(the reversion prompt uses this list), and the problems and warnings textcheck found.
06_count.txt is the same check as a plain-text report you can read.
"""
from core import textcheck

SOURCE_FILE = "05_winner.txt"
JSON_FILE = "06_count.json"
REPORT_FILE = "06_count.txt"


def count_text(text, allowed, target, tolerance, expected_dnt=None):
    """The count result for one translation (a dict that is saved as 06_count.json), and the textcheck result."""
    r = textcheck.check(text, allowed, target, expected_dnt=expected_dnt)
    new = len(r["unknown"])
    result = {
        "new_chars_count": new,
        "target": target,
        "tolerance": tolerance,
        "over_by": max(0, new - target),
        "revert_threshold": target + tolerance,
        "revert_needed": new > target + tolerance,
        "total_chars": r["total_chars"],
        "unique_chars": r["unique"],
        "percent": r["percent"],
        "new_chars": r["new_details"],
        "problems": r["problems"],
        "warnings": r["warnings"],
    }
    return result, r


def run(cfg, job):
    text = job.read(SOURCE_FILE)
    allowed = textcheck.load_allowed(cfg)
    target = textcheck.new_char_target(cfg)
    tolerance = textcheck.revert_tolerance(cfg)
    tagged_ok = job.exists("03_tagged.txt") and job.status().get("meta", {}).get("tagging") == "ok"
    expected = textcheck.extract_dnt(job.read("03_tagged.txt")) if tagged_ok else None

    result, r = count_text(text, allowed, target, tolerance, expected)
    job.write_json(JSON_FILE, result)
    job.write(REPORT_FILE, textcheck.format_report(r) + "\n")

    n = result["new_chars_count"]
    note = f"{n} new characters (target {target}, reversion above {result['revert_threshold']}): "
    if result["revert_needed"]:
        note += f"{n - target} over the target, so English reversion will run"
    elif n > target:
        note += f"{n - target} over the target but within the tolerance, so no reversion"
    else:
        note += "within the target, so no reversion"
    if result["problems"]:
        note += "; WARNING: " + "; ".join(result["problems"])
    return note
