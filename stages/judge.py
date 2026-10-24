"""Stages 5 and 7b, judge: a model (any profile you choose in config.json) picks the best of several candidates.

    judge_translation   picks the best of the three translations (stage 5, below)
    judge_comprehension picks the most comprehensible English reversion (stage 7b, at the end of this file)

    python3 run.py judge_translation --job ID     # reads 04_meta.json + 04_translation_N.txt
                                                  # writes 05_winner.txt, 05_judge.json

The candidates are shown with their <dnt> tags removed, in a shuffled order under the labels A, B, C, so
position and temperature cannot sway the judge. Each one carries its count of characters outside the
learner's known list. The prompt is prompts/judge_translation.txt; the rubric is prompts/judge_rubric.txt
(edit it freely). The model must end with "WINNER: X". An unreadable reply is retried (stages.judge_translation.retries times, each retry a little warmer by
retry_temp_step); if that fails
too, the middle-temperature candidate wins and the note says so. With only one candidate no call is made.

05_winner.txt keeps the <dnt> tags (the counter needs them); package.py removes them.
"""
import random
import re

from core import llm, schema, tags, textcheck
from core.config import ConfigError
from core.prompts import load_prompt

META_FILE = "04_meta.json"
WINNER_FILE = "05_winner.txt"
JUDGE_FILE = "05_judge.json"
REVERT_META_FILE = "07_meta.json"
FINAL_FILE = "08_winner.txt"
FINAL_JUDGE_FILE = "08_judge.json"
LABELS = "ABCDEFGH"
ENGLISH_FILES = ("03_tagged.txt", "02_simplified.txt", "01_clean.txt")


def shuffled(items, seed):
    """A new list in a shuffled order; the same seed always gives the same order."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def parse_winner(reply, count):
    """The index (0-based) of the letter after the last 'WINNER:' in the reply, or None."""
    letters = LABELS[:count]
    found = re.findall(r"^[ \t>*_#-]*WINNER[ \t]*:[ \t*_\[(\"']*([A-Za-z])\b", reply, flags=re.M | re.I)
    if found and found[-1].upper() in letters:
        return letters.index(found[-1].upper())
    return None


def parse_reason(reply):
    m = re.search(r"^[ \t>*_#-]*REASON[ \t]*:[ \t*_]*(.+?)(?=^[ \t>*_#-]*WINNER|\Z)", reply, flags=re.M | re.S | re.I)
    return " ".join(m.group(1).split()) if m else ""


def choose(cfg, job, build_prompt, texts, seed, stage):
    """Ask the cloud judge to pick one of `texts`. Returns (index, reason, how, reply).

    build_prompt(labelled) gets [(label, index into texts), ...] in the shuffled order and returns the prompt.
    how is "judge", or "fallback" when the reply could not be read (index is then the middle one).
    The model, temperature, token limit and number of retries come from the stage's settings."""
    opts = schema.stage_opts(cfg, stage)
    order = shuffled(range(len(texts)), seed)                 # order[k] = which text is shown under label k
    prompt = build_prompt([(LABELS[k], i) for k, i in enumerate(order)])
    last = ""
    for attempt in range(1, opts["retries"] + 2):
        try:
            reply = llm.chat(cfg, prompt, model=opts["model"], temperature=schema.attempt_temperature(cfg, opts, attempt - 1),
                             max_tokens=opts["max_tokens_override"])
        except ConfigError:
            raise                                              # a settings problem (no key, no model): stop at once
        except RuntimeError as e:                              # unreachable or an HTTP error
            last = f"call failed: {e}"
            job.log(f"judge: attempt {attempt} {last}")
            continue
        pick = parse_winner(reply, len(texts))
        if pick is not None:
            return order[pick], parse_reason(reply), "judge", reply
        last = reply
        job.log(f"judge: attempt {attempt} had no readable WINNER line")
    return len(texts) // 2, "", "fallback", last


def run_translation(cfg, job, use_judge=True):
    meta = job.read_json(META_FILE)
    good = [c for c in meta["candidates"] if c.get("file") and job.exists(c["file"])]
    if not good:
        raise RuntimeError(f"{META_FILE} lists no translations. Run: python3 run.py translate --job {job.id}")
    good.sort(key=lambda c: c["n"])
    texts = [job.read(c["file"]).strip() for c in good]

    if len(good) == 1:
        winner, reason, how, reply = 0, "", "only", ""
    elif not use_judge:
        winner, reason, how, reply = len(good) // 2, "", "skipped", ""
    else:
        english = next(textcheck.strip_dnt_tags(job.read(f)).strip() for f in ENGLISH_FILES if job.exists(f))
        template = load_prompt(cfg, "judge_translation.txt")
        rubric = load_prompt(cfg, "judge_rubric.txt")
        if "{{CANDIDATES}}" not in template or "{{ENGLISH}}" not in template:
            raise ConfigError("prompts/judge_translation.txt must contain {{ENGLISH}} and {{CANDIDATES}}.")

        def build(labelled):
            blocks = []
            for label, i in labelled:
                blocks.append(f"CANDIDATE {label} (characters outside the learner's known list: "
                              f"{good[i]['new_chars']}):\n" + textcheck.strip_dnt_tags(texts[i]))
            return (template.replace("{{RUBRIC}}", rubric).replace("{{CANDIDATES}}", "\n\n".join(blocks))
                            .replace("{{ENGLISH}}", english))

        winner, reason, how, reply = choose(cfg, job, build, texts, seed=job.id, stage="judge_translation")

    chosen = good[winner]
    job.write(WINNER_FILE, texts[winner] + "\n")
    job.write_json(JUDGE_FILE, {"winner": chosen["n"], "temperature": chosen["temperature"], "method": how,
                                "reason": reason, "candidates": [c["n"] for c in good], "reply": reply})
    note = f"winner: translation {chosen['n']} (t={chosen['temperature']:g}, {chosen['new_chars']} new characters)"
    if how == "only":
        note += "; the only translation, so no judge call"
    elif how == "skipped":
        note += "; the judge is turned off, so the middle temperature was used"
    elif how == "fallback":
        note += "; WARNING: the judge's answer could not be read, so the middle temperature was used"
    elif reason:
        note += "; " + reason
    return note


def skip_translation(cfg, job):
    """The judge turned off: the middle-temperature translation becomes 05_winner.txt, with no model call."""
    return run_translation(cfg, job, use_judge=False)


def skip_comprehension(cfg, job):
    """The judge turned off: the middle valid reversion becomes 08_winner.txt, with no model call."""
    return run_comprehension(cfg, job, use_judge=False)


def run_comprehension(cfg, job, use_judge=True):
    """Stage 7b: pick the most comprehensible of the VALID reversions. Writes 08_winner.txt and 08_judge.json.

    prompts/judge_comprehension.txt and judge_comprehension_rubric.txt are used. With one valid reversion it
    wins without a call; with none, 05_winner.txt (the translation as it was) is kept. The reader is
    assumed to know English and some Chinese, so this judge looks at flow and meaning, not just at counts."""
    meta = job.read_json(REVERT_META_FILE)
    good = sorted((c for c in meta["candidates"] if c.get("valid") and c.get("file") and job.exists(c["file"])),
                  key=lambda c: c["n"])
    if not good:
        job.write(FINAL_FILE, job.read(WINNER_FILE).strip() + "\n")
        job.write_json(FINAL_JUDGE_FILE, {"winner": None, "method": "kept_translation", "reason": "", "reply": ""})
        return "no valid reversion, so the translation from 05_winner.txt is kept as it is"
    texts = [job.read(c["file"]).strip() for c in good]

    if len(good) == 1:
        winner, reason, how, reply = 0, "", "only", ""
    elif not use_judge:
        winner, reason, how, reply = len(good) // 2, "", "skipped", ""
    else:
        english = next(textcheck.strip_dnt_tags(job.read(f)).strip() for f in ENGLISH_FILES if job.exists(f))
        template = load_prompt(cfg, "judge_comprehension.txt")
        rubric = load_prompt(cfg, "judge_comprehension_rubric.txt")
        if "{{CANDIDATES}}" not in template or "{{ENGLISH}}" not in template:
            raise ConfigError("prompts/judge_comprehension.txt must contain {{ENGLISH}} and {{CANDIDATES}}.")

        def build(labelled):
            blocks = []
            for label, i in labelled:
                blocks.append(f"CANDIDATE {label} (English words put back: {good[i]['en_spans']}; characters outside "
                              f"the learner's known list: {good[i]['new_chars']}):\n" + tags.strip_all_tags(texts[i]))
            return (template.replace("{{RUBRIC}}", rubric).replace("{{CANDIDATES}}", "\n\n".join(blocks))
                            .replace("{{ENGLISH}}", english))

        winner, reason, how, reply = choose(cfg, job, build, texts, seed=job.id + "-comprehension",
                                         stage="judge_comprehension")

    chosen = good[winner]
    job.write(FINAL_FILE, texts[winner] + "\n")
    job.write_json(FINAL_JUDGE_FILE, {"winner": chosen["n"], "temperature": chosen["temperature"], "method": how,
                                      "reason": reason, "candidates": [c["n"] for c in good], "reply": reply})
    note = (f"winner: reversion {chosen['n']} (t={chosen['temperature']:g}, {chosen['new_chars']} new characters, "
            f"{chosen['en_spans']} English insertions)")
    if how == "only":
        note += "; the only valid reversion, so no judge call"
    elif how == "skipped":
        note += "; the judge is turned off, so the middle temperature was used"
    elif how == "fallback":
        note += "; WARNING: the judge's answer could not be read, so the middle temperature was used"
    elif reason:
        note += "; " + reason
    return note
