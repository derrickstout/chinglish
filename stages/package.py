"""Stage 8, package: the finished result, ready to read or to email. No model is used.

    python3 run.py package --job ID         # writes 09_result.txt and 09_result.json

The text that is packaged is 08_winner.txt (the comprehension judge's pick) when reversion ran, and
05_winner.txt (the judge's pick of the translations) when it did not. This is the one place the <dnt> and
<en> tags are removed: the reader sees plain text.

09_result.txt has four parts: TRANSLATION, ENGLISH (the simplified English the translation follows),
ORIGINAL ARTICLE (the cleaned source), and NOTES (how the result came about, the new-character count against
the target, the new characters left, and anything that went wrong or was cut). 09_result.json holds the same
parts for the email stage to use.
"""
import datetime

from core import tags, textcheck
from core.config import resolve_path

RESULT_TXT = "09_result.txt"
RESULT_JSON = "09_result.json"
STALE_DAYS = 45                       # allowed_chars.json older than this gets a "re-export from Pleco" hint


def final_choice(job):
    """(file name, 'reversion' or 'translation') of the text to package."""
    try:
        needed = bool(job.read_json("06_count.json").get("revert_needed"))
    except (FileNotFoundError, ValueError):
        needed = False
    if needed and job.exists("08_winner.txt"):
        return "08_winner.txt", "reversion"
    return "05_winner.txt", "translation"


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _char_list(chars, counts):
    return " ".join(c if counts.get(c, 1) == 1 else f"{c}x{counts[c]}" for c in chars)


def build_notes(cfg, job, final_file, kind, check, target, tolerance):
    """The NOTES section as a list of lines, plus the list of warnings (also returned separately)."""
    meta = job.status().get("meta", {})
    lines, warnings = [], []
    new = len(check["unknown"])
    stages = job.status().get("stages", {})
    off = [n for n, e in stages.items() if e.get("status") == "skipped" and e.get("note", "").startswith("turned off")]
    if off:
        lines.append("Steps turned off in config.json: " + ", ".join(off) + ".")

    # how the translation was chosen
    if job.exists("05_judge.json"):
        j = job.read_json("05_judge.json")
        how = {"judge": "chosen by the cloud judge", "only": "the only translation made",
               "fallback": "chosen by default (the judge's answer could not be read)",
               "skipped": "picked without a judge (the judge is turned off)"}.get(j["method"], j["method"])
        line = f"Translation {j['winner']} of {len(j['candidates'])} (temperature {j['temperature']:g}), {how}."
        if j.get("reason"):
            line += f" Judge: {j['reason']}"
        lines.append(line)
        if j["method"] == "fallback":
            warnings.append("the translation judge's answer could not be read; the middle temperature was used")
    count = job.read_json("06_count.json") if job.exists("06_count.json") else None

    # new characters
    verdict = "within the target" if new <= target else (
        f"{new - target} over the target" + (", within the tolerance" if new <= target + tolerance else ""))
    lines.append(f"New characters in this text: {new} (target {target}); {verdict}.")
    if new > target + tolerance:
        warnings.append(f"still {new - target} over the new-character target of {target}")

    if kind == "reversion":
        before = count["new_chars_count"] if count else "?"
        r = job.read_json("08_judge.json") if job.exists("08_judge.json") else {}
        how = " (the judge is turned off, so the middle temperature was used)" if r.get("method") == "skipped" else ""
        rm = job.read_json("07_meta.json") if job.exists("07_meta.json") else {"candidates": []}
        pick = next((c for c in rm["candidates"] if c["n"] == r.get("winner")), {})
        lines.append(f"English reversion ran: {before} new characters before, {new} after, with "
                     f"{_plural(pick.get('en_spans', len(tags.extract_en(job.read(final_file)))), 'English insertion')}"
                     f" (reversion {r.get('winner')}, temperature {pick.get('temperature', 0):g}"
                     + (f"; judge: {r['reason']}" if r.get("reason") else "") + ").")
        if r.get("method") == "fallback":
            warnings.append("the comprehension judge's answer could not be read; the middle temperature was used")
    elif count and count.get("revert_needed") and "revert" in off:
        lines.append("English reversion would have been used, but it is turned off, so the translation is unchanged.")
    elif count and count.get("revert_needed"):
        lines.append("English reversion ran, but no attempt passed the checks, so the translation is unchanged.")
        warnings.append("English reversion produced nothing usable")
    elif count and new > target:
        lines.append("English reversion was not needed (the excess was within the tolerance of "
                     f"{tolerance}).")

    # the new characters themselves
    counts = {d["char"]: d["count"] for d in check["new_details"]}
    if check["in_pool"]:
        lines.append(f"New characters from your preferred list ({len(check['in_pool'])}): "
                     + _char_list(check["in_pool"], counts))
    if check["outside_pool"]:
        lines.append(f"New characters outside your preferred list ({len(check['outside_pool'])}): "
                     + _char_list(check["outside_pool"], counts))

    # things that went wrong or were cut
    if meta.get("simplification") == "failed":
        warnings.append("the simplifying step failed, so the original wording was translated without being simplified")
    if meta.get("tagging") == "discarded":
        warnings.append("names could not be protected (the tagging was thrown away), so some may be translated: "
                        + str(meta.get("tagging_problem", "")))
    cut = meta.get("ingest_cut_to")
    if cut is None:
        note = job.status().get("stages", {}).get("ingest", {}).get("note", "")
        cut = 1 if "CUT TO THE FIRST" in note else 0
    if cut:
        warnings.append("the article was longer than stages.ingest.max_chars, so only the beginning was translated"
                        + (f" (the first {cut:,} characters)" if cut > 1 else ""))
    for problem in check["problems"]:
        warnings.append(problem)
    for w in check["warnings"]:
        warnings.append(w)
    if job.exists("04_meta.json"):
        won = job.read_json("05_judge.json")["winner"] if job.exists("05_judge.json") else None
        for c in job.read_json("04_meta.json")["candidates"]:
            if c["n"] == won and c.get("notes"):
                lines.append("The model's note on the translation: " + " ".join(c["notes"].split()))

    # the character list
    path = resolve_path(cfg, "data") / "allowed_chars.json"
    if path.exists():
        age = (datetime.date.today() - datetime.date.fromtimestamp(path.stat().st_mtime)).days
        lines.append(f"Character list (allowed_chars.json) last updated {age} day{'s' if age != 1 else ''} ago."
                     + (" Re-export from Pleco and run data/pleco_import.py to refresh it." if age > STALE_DAYS else ""))
    return lines, warnings


def run(cfg, job):
    final_file, kind = final_choice(job)
    final = job.read(final_file).strip()
    allowed = textcheck.load_allowed(cfg)
    target = textcheck.new_char_target(cfg)
    tolerance = textcheck.revert_tolerance(cfg)
    tagged_ok = job.exists("03_tagged.txt") and job.status().get("meta", {}).get("tagging") == "ok"
    expected = textcheck.extract_dnt(job.read("03_tagged.txt")) if tagged_ok else None
    check = textcheck.check(tags.drop_en_text(final), allowed, target, expected_dnt=expected)

    translation = tags.strip_all_tags(final).strip()
    english = job.read("02_simplified.txt").strip() if job.exists("02_simplified.txt") else ""
    if job.status().get("meta", {}).get("simplification") in ("skipped", "failed"):
        english = "(not simplified)"                          # 02 is only a copy of the original here
    original = job.read("01_clean.txt").strip()
    lines, warnings = build_notes(cfg, job, final_file, kind, check, target, tolerance)

    notes = list(lines)
    if warnings:
        notes = ["WARNINGS:"] + ["  - " + w for w in warnings] + [""] + notes
    text = "\n".join(["TRANSLATION", "", translation, "", "-" * 40, "",
                      "ENGLISH (simplified)", "", english, "", "-" * 40, "",
                      "ORIGINAL ARTICLE", "", original, "", "-" * 40, "",
                      "NOTES", ""] + notes) + "\n"
    job.write(RESULT_TXT, text)
    job.write_json(RESULT_JSON, {"translation": translation, "english": english, "original": original,
                                 "notes": notes, "warnings": warnings, "source": final_file, "kind": kind,
                                 "new_chars": len(check["unknown"]), "target": target})
    out = f"packaged {final_file} ({len(check['unknown'])} new characters, target {target})"
    if warnings:
        out += "; WARNING: " + "; ".join(warnings)
    return out
