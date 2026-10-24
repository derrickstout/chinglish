"""Stage 7, revert: put English back in where the winning translation has too many new characters.

    python3 run.py revert --job ID      # reads 05_winner.txt, 06_count.json (runs only if revert_needed)
                                        # writes 07_reversion_1..3.txt, 07_meta.json

The local model gets prompts/revert.txt, filled in with the original English, the winning translation,
and the list of new characters (from 06_count.json) with how often each occurs and where. It is asked to
replace Chinese words that contain new characters with the original English, wrapped in <en>...</en>,
preferring nouns and simple verbs. It makes stages.revert.passes attempts, starting at its temperature and rising by pass_temp_step
(0.3, 0.6, 0.9 by default).

Every attempt is checked by this script, not trusted. An attempt is VALID only if:
  - its <en> tags are balanced,
  - every <en> span is English that appears in the original English,
  - every <dnt> span from the tagged story is still there (when the story was tagged),
  - it still has Chinese in it, and
  - it has fewer new characters than the winning translation.
Invalid attempts are still saved (and the reasons recorded in 07_meta.json) so you can read them, but only
valid ones go on to the comprehension judge. If none is valid the judge keeps the winning translation.
"""
import re
import time

from core import llm, schema, tags, textcheck
from core.config import ConfigError
from core.prompts import load_prompt, strip_wrapper, token_budget

META_FILE = "07_meta.json"
PLACEHOLDERS = ("{{ENGLISH}}", "{{TRANSLATION}}", "{{NEW_CHARS}}", "{{TARGET}}")


def format_new_chars(details):
    """One line per new character: the character, how often it occurs, whether it is preferred, and a snippet."""
    lines = []
    ordered = sorted(details, key=lambda d: (d["in_pool"], -d["count"]))        # not-preferred ones first
    for d in ordered:
        where = "" if d["in_pool"] else "  [not on the preferred list]"
        times = "once" if d["count"] == 1 else f"{d['count']} times"
        lines.append(f"{d['char']}  ({times}){where}  e.g. \"{d['context']}\"")
    return "\n".join(lines) if lines else "(none)"


def build_prompt(template, english, translation, details, target):
    """Fill the template. Raises ConfigError if it has lost a placeholder."""
    missing = [p for p in PLACEHOLDERS if p not in template]
    if missing:
        raise ConfigError("prompts/revert.txt is missing " + ", ".join(missing) + ".")
    prompt = template.replace("{{TARGET}}", str(target)).replace("{{NEW_CHARS}}", format_new_chars(details))
    # The two texts go in last, so text inside them that looks like a placeholder is never replaced.
    return prompt.replace("{{ENGLISH}}", english).replace("{{TRANSLATION}}", translation)


def _norm(text):
    return " ".join(text.lower().split())


def verify(body, english, expected_dnt, before, allowed, target):
    """Check one reversion attempt. Returns (problems, warnings, new_chars_count).

    english is the original English with its tags removed; before is the winner's new-character count."""
    problems = []
    if not tags.balanced_en(body):
        problems.append("the <en> tags were unbalanced")
    original = _norm(english)
    foreign = [s for s in tags.extract_en(body) if not s or _norm(textcheck.strip_dnt_tags(s)) not in original]
    if foreign:
        problems.append("English that is not in the original: " + ", ".join(f'"{s}"' for s in foreign[:5])
                        + (" ..." if len(foreign) > 5 else ""))
    r = textcheck.check(tags.drop_en_text(body), allowed, target, expected_dnt=expected_dnt)
    problems += r["problems"]
    new = len(r["unknown"])
    if new >= before:
        problems.append(f"the new-character count did not drop ({new}, was {before})")
    return problems, r["warnings"], new


def run(cfg, job):
    winner = job.read("05_winner.txt").strip()
    counted = job.read_json("06_count.json")
    tagged = job.read("03_tagged.txt").strip()
    template = load_prompt(cfg, "revert.txt")
    allowed = textcheck.load_allowed(cfg)
    target = textcheck.new_char_target(cfg)
    opts = schema.stage_opts(cfg, "revert")
    temps = schema.pass_temperatures(opts)
    before = counted["new_chars_count"]
    english = textcheck.strip_dnt_tags(tagged)
    expected = textcheck.extract_dnt(tagged) if job.status().get("meta", {}).get("tagging") == "ok" else None

    prompt = build_prompt(template, tagged, winner, counted["new_chars"], target)
    job.log(f"revert: {before} new characters in the winner (target {target}); temperatures {temps}")
    for old in job.path.glob("07_reversion_*.txt"):
        old.unlink()

    llm.ensure_model_switched(cfg, opts["model"])
    limit = token_budget(cfg, "revert", winner, factor=1.5, extra=800)
    candidates = []
    for n, temp in enumerate(temps, 1):
        entry = {"n": n, "temperature": temp, "file": None, "valid": False, "error": None}
        started = time.time()
        try:
            reply = llm.chat(cfg, prompt, model=opts["model"], temperature=temp, max_tokens=limit)
            body, _ = strip_wrapper(reply)
            body, notes = textcheck.split_reply(body)
            if not body:
                raise RuntimeError("the reply was empty")
            name = f"07_reversion_{n}.txt"
            job.write(name, body + "\n")
            problems, warnings, new = verify(body, english, expected, before, allowed, target)
            entry.update(file=name, valid=not problems, problems=problems, warnings=warnings, new_chars=new,
                         en_spans=len(tags.extract_en(body)), notes=notes)
        except ConfigError:
            raise
        except (RuntimeError, OSError) as e:
            entry["error"] = str(e)
            job.log(f"revert: attempt {n} (temperature {temp}) failed: {e}")
        entry["seconds"] = round(time.time() - started, 1)
        candidates.append(entry)

    job.write_json(META_FILE, {"temperatures": temps, "target_new_chars": target, "winner_new_chars": before,
                               "candidates": candidates})
    made = [c for c in candidates if c["file"]]
    if not made:
        raise RuntimeError(f"All {len(candidates)} reversion attempts failed. Last error: {candidates[-1]['error']}")

    valid = [c for c in candidates if c["valid"]]
    parts = [f"#{c['n']} t={c['temperature']:g}: "
             + ("FAILED" if not c["file"] else f"{c['new_chars']} new" + ("" if c["valid"] else " (rejected)"))
             for c in candidates]
    note = f"{len(valid)} of {len(candidates)} valid (winner had {before}); " + "; ".join(parts)
    if not valid:
        note += "; WARNING: none passed the checks, so the translation will be kept as it is (reasons in " + META_FILE + ")"
    return note
