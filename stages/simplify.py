"""Stage 2, simplify: rewrite the cleaned article in simpler English with the local model.

    python3 run.py simplify --job ID        # reads 01_clean.txt, writes 02_simplified.txt

The prompt is prompts/simplify.txt, sent as the system message; the article is the user message. The reply
is tidied (a wrapping code fence, an introduction line and markdown marks removed) and checked for signs
of trouble. Trouble is reported as a WARNING in the stage note and in log.txt; it never stops the stage,
because a slightly odd simplification is still worth reading.

An empty reply, or a call that fails, is tried again up to stages.simplify.retries more times, each time
at a temperature raised by retry_temp_step (never above max_temperature). If every reply is empty,
stages.simplify.on_fail decides: "stop" (the default) fails the stage, "continue" passes the text on as it
was. If no attempt ever got a reply (the server is down), the stage always fails.

The reply's token limit is raised for a long article (core.prompts.token_budget), so a long story is not
cut off by the model's max_tokens.
"""
import re

from core import llm, schema
from core.config import ConfigError
from core.prompts import load_prompt, strip_wrapper, token_budget

SOURCE_FILE = "01_clean.txt"
OUTPUT_FILE = "02_simplified.txt"

_CHINESE = re.compile(r"[一-鿿]")


def tidy(reply):
    """The reply without a wrapper (fence, introduction line), bold or heading marks, or runs of blank
    lines. Returns (text, notes); notes says what was removed."""
    text, notes = strip_wrapper(reply)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), notes


def check_reply(source, simplified):
    """Warnings (a list of short sentences) about a simplified text. An empty list means nothing looks wrong."""
    warnings = []
    before, after = len(source.split()), len(simplified.split())
    if before >= 40 and after < 0.5 * before:
        warnings.append(f"the simplified text has only {round(100 * after / before)}% of the original words; "
                        "it may be a summary, or the model ran out of tokens (max_tokens)")
    elif before >= 40 and after > 1.6 * before:
        warnings.append(f"the simplified text is {round(100 * after / before)}% of the original length")
    if simplified and not simplified.rstrip().endswith(tuple(".!?\"'”’)")):
        warnings.append("the text does not end with a full stop; the model may have been cut off "
                        "(max_tokens)")
    if _CHINESE.search(simplified):
        warnings.append("the text contains Chinese characters; the model may have translated it")
    return warnings


def run(cfg, job):
    source = job.read(SOURCE_FILE).strip()
    prompt = load_prompt(cfg, "simplify.txt")
    opts = schema.stage_opts(cfg, "simplify")
    llm.ensure_model_switched(cfg, opts["model"])
    limit = token_budget(cfg, "simplify", source)

    attempts = opts["retries"] + 1
    simplified, removed, error, got_reply = "", [], None, False
    for n in range(attempts):
        temp = schema.attempt_temperature(cfg, opts, n)
        try:
            reply = llm.chat(cfg, prompt, source, model=opts["model"], temperature=temp, max_tokens=limit)
        except ConfigError:
            raise                                                   # a settings problem: retrying cannot help
        except RuntimeError as e:
            error = e
            job.log(f"simplify: attempt {n + 1} of {attempts} (temperature {temp}) failed: {e}")
            continue
        got_reply = True
        simplified, removed = tidy(reply)
        if simplified:
            break
        job.log(f"simplify: attempt {n + 1} of {attempts} (temperature {temp}) gave an empty reply")

    if not simplified:
        if not got_reply:
            raise error                                             # no reply at all is always an error
        if opts["on_fail"] == "stop":
            raise RuntimeError(f"Simplifying failed: the model's reply was empty after {attempts} "
                               f"attempt{'s' if attempts != 1 else ''}. Set stages.simplify.on_fail to \"continue\" "
                               "to go on with the text as it is.")
        job.write(OUTPUT_FILE, source + "\n")
        job.set_meta(simplification="failed")
        job.log(f"simplify: gave up after {attempts} attempts; the text is passed on as it was")
        return (f"WARNING: simplifying failed after {attempts} attempt{'s' if attempts != 1 else ''} "
                "(empty replies), so the text is translated as it is")
    job.write(OUTPUT_FILE, simplified + "\n")
    job.set_meta(simplification="ok")
    note = f"{len(source.split()):,} -> {len(simplified.split()):,} words"
    if n:
        note += f" (attempt {n + 1} of {attempts})"
    if removed:
        note += "; " + ", ".join(removed)
    warnings = check_reply(source, simplified)
    for w in warnings:
        job.log("simplify warning: " + w)
    if warnings:
        note += "; WARNING: " + "; ".join(warnings)
    return note


def skip(cfg, job):
    """stages.simplify.enabled is false: the cleaned article goes on as it is."""
    job.write(OUTPUT_FILE, job.read(SOURCE_FILE).strip() + "\n")
    job.set_meta(simplification="skipped")
    return "turned off; the article is passed on as it is"
