"""Stage 3, tag: wrap the text that must stay in English (names, acronyms, quotes) in <dnt>...</dnt>.

    python3 run.py tag --job ID             # reads 02_simplified.txt, writes 03_tagged.txt

The prompt is prompts/dnt.txt with the story added after it, sent as one user message. The model's reply
is accepted only if every tag is balanced and the story's words are unchanged once the tags are removed.

A rejected reply is asked for again, up to stages.tag.retries more times. Each retry adds prompts/dnt_retry.txt
(which says why the last reply was rejected) and raises the temperature by stages.tag.retry_temp_step, never
above max_temperature. A call that fails outright (the server is down) is retried the same way, without the
reason. If every attempt is rejected, stages.tag.on_fail decides: "continue" (the default) throws the tagging
away: 03_tagged.txt then holds the plain story, the last raw reply is kept in 03_tag_reply_raw.txt, and
job.json records tagging = "discarded", so later stages translate the story without protected text instead
of working from a damaged one; "stop" fails the stage. If no attempt ever got a reply, the stage always fails.
The story itself can never be corrupted by this stage.

job.json (meta) records tagging = "ok" or "discarded" and how many attempts were used; later stages use it
to know whether the <dnt> spans can be checked.
"""
import re

from core import llm, schema, textcheck
from core.config import ConfigError
from core.prompts import load_prompt, strip_wrapper, token_budget

SOURCE_FILE = "02_simplified.txt"
OUTPUT_FILE = "03_tagged.txt"
RAW_FILE = "03_tag_reply_raw.txt"


def check_tagging(original, reply):
    """Judge a tagging reply against the story it was given. Returns (tagged text, problem).

    On success the tagged text comes back with an empty problem; on failure the tagged text is None and
    the problem is a short phrase ("the tags were unbalanced", "the model changed the wording")."""
    tagged, _ = strip_wrapper(reply)                                  # a code fence or "Here is ..." line
    tagged = re.sub(r"<dnt>\s*</dnt>", "", tagged)                    # empty pairs carry no information
    pairs = len(textcheck.DNT_SPAN.findall(tagged))
    if not (tagged.count("<dnt>") == tagged.count("</dnt>") == pairs):
        return None, "the tags were unbalanced"
    if " ".join(textcheck.strip_dnt_tags(tagged).split()) != " ".join(original.split()):
        return None, "the model changed the wording"
    return tagged, ""


def _prompt(template, story, problem=None, retry_template=None):
    """The message for one attempt: the tagging prompt and story, plus (after a rejected reply) why it was rejected."""
    message = template + "\n\n" + story
    if problem and retry_template:
        message += "\n\n" + retry_template.replace("{{PROBLEM}}", problem)
    return message


def run(cfg, job):
    story = job.read(SOURCE_FILE).strip()
    template = load_prompt(cfg, "dnt.txt")
    retry_template = load_prompt(cfg, "dnt_retry.txt")
    opts = schema.stage_opts(cfg, "tag")
    llm.ensure_model_switched(cfg, opts["model"])
    limit = token_budget(cfg, "tag", story, factor=1.4)

    problem, reply, error = "", None, None        # what went wrong last, the last reply received, the last call error
    attempts = opts["retries"] + 1
    for n in range(attempts):
        temp = schema.attempt_temperature(cfg, opts, n)
        try:
            reply = llm.chat(cfg, _prompt(template, story, problem, retry_template), model=opts["model"],
                             temperature=temp, max_tokens=limit)
        except ConfigError:
            raise                                                   # a settings problem: retrying cannot help
        except RuntimeError as e:
            error = e
            job.log(f"tag: attempt {n + 1} of {attempts} (temperature {temp}) failed: {e}")
            continue
        tagged, problem = check_tagging(story, reply)
        if tagged is not None:
            break
        job.log(f"tag: attempt {n + 1} of {attempts} (temperature {temp}) was rejected: {problem}")
    else:
        tagged = None

    if tagged is None and reply is None:                            # no reply at all: a broken connection, not bad tagging
        raise error
    if tagged is None:
        if opts["on_fail"] == "stop":
            job.write(RAW_FILE, reply + "\n")
            raise RuntimeError(f"Tagging failed after {attempts} attempt{'s' if attempts != 1 else ''} ({problem}); "
                               f"the last reply is in {RAW_FILE}. Set stages.tag.on_fail to \"continue\" to go on without tags.")
        job.write(OUTPUT_FILE, story + "\n")
        job.write(RAW_FILE, reply + "\n")
        job.set_meta(tagging="discarded", tagging_problem=problem, tagging_attempts=attempts)
        job.log(f"tag: discarded after {attempts} attempts ({problem}); the last reply is saved as {RAW_FILE}")
        return (f"WARNING: tagging discarded ({problem}) after {attempts} attempt{'s' if attempts != 1 else ''}; "
                f"the story will be translated without <dnt> tags (the last reply is in {RAW_FILE})")

    job.file(RAW_FILE).unlink(missing_ok=True)                       # left over from an earlier failed run
    job.write(OUTPUT_FILE, tagged + "\n")
    spans = textcheck.extract_dnt(tagged)
    used = n + 1
    job.set_meta(tagging="ok", tagging_problem="", protected_spans=len(spans), tagging_attempts=used)
    extra = f" (attempt {used} of {attempts})" if used > 1 else ""
    if not spans:
        return "no text needed protecting" + extra
    shown = ", ".join(spans[:8]) + (" ..." if len(spans) > 8 else "")
    return f"protected {len(spans)} span{'s' if len(spans) != 1 else ''}: {shown}{extra}"


def skip(cfg, job):
    """stages.tag.enabled is false: the story goes on with no <dnt> tags."""
    job.write(OUTPUT_FILE, job.read(SOURCE_FILE).strip() + "\n")
    job.file(RAW_FILE).unlink(missing_ok=True)
    job.set_meta(tagging="skipped", tagging_problem="", protected_spans=0)
    return "turned off; the story goes on with nothing protected"
