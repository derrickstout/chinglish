"""Stage 4, translate: three translations of the tagged story into Traditional Chinese, at rising temperatures.

    python3 run.py translate --job ID       # reads 03_tagged.txt, writes 04_translation_1..3.txt, 04_meta.json

The prompt is prompts/translate.txt with four placeholders filled in:
    {{STORY}}            the tagged story (03_tagged.txt)
    {{KNOWN_CHARS}}      the Mastered and Learning characters from allowed_chars.json
    {{PREFERRED_CHARS}}  the "unknown" pool in allowed_chars.json (new characters to use first)
    {{TARGET}}           translation.target_new_chars (a SOFT target: accuracy comes first)
It is sent as one user message, once per pass: stages.translate.passes passes, starting at its temperature and rising by pass_temp_step
(0.3, 0.6, 0.9 by default).

The <dnt> tags stay in the saved translations (the judge and the counter need them; package.py removes
them). Each translation is checked with core.textcheck, and the numbers are recorded in 04_meta.json for
the judge and for you; a problem there never stops the stage. A call that fails (or returns nothing) is
recorded and the other temperatures still run. If every call fails the stage fails.

Files: 04_translation_N.txt (N = 1, 2, 3 by temperature order), 04_meta.json.
"""
import re
import time

from core import llm, schema, textcheck
from core.config import ConfigError
from core.prompts import load_prompt, token_budget

SOURCE_FILE = "03_tagged.txt"
META_FILE = "04_meta.json"
PLACEHOLDERS = ("{{STORY}}", "{{KNOWN_CHARS}}", "{{PREFERRED_CHARS}}", "{{TARGET}}")


def char_lists(allowed):
    """(known, preferred) as strings: Mastered+Learning, and the pool characters that are not yet known."""
    known = list(dict.fromkeys(list(allowed["mastered"]) + list(allowed["learning"])))
    known_set = set(known)
    preferred = [u["char"] for u in allowed["unknown"] if u["char"] not in known_set]
    return "".join(known), "".join(dict.fromkeys(preferred))


def build_prompt(template, story, allowed, target):
    """Fill the template. Raises ConfigError if the template has lost a placeholder."""
    missing = [p for p in PLACEHOLDERS if p not in template]
    if missing:
        raise ConfigError("prompts/translate.txt is missing " + ", ".join(missing) + ".")
    known, preferred = char_lists(allowed)
    # The story goes in last so that text inside it that looks like a placeholder is never replaced.
    prompt = (template.replace("{{KNOWN_CHARS}}", known or "(none)")
                      .replace("{{PREFERRED_CHARS}}", preferred or "(none)")
                      .replace("{{TARGET}}", str(target)))
    return prompt.replace("{{STORY}}", story)


def estimate_tokens(text):
    """A rough token count: Chinese characters about 1.5 each, anything else about 4 characters a token."""
    han = sum(1 for c in text if textcheck.is_han(c))
    return int(han * 1.5 + (len(text) - han) / 4)


def summarise(result):
    """The few numbers worth keeping from textcheck.check, as plain JSON."""
    return {"new_chars": len(result["unknown"]), "outside_pool": len(result["outside_pool"]),
            "unique_chars": result["unique"], "problems": result["problems"],
            "warnings": result["warnings"], "notes": result["notes"]}


def run(cfg, job):
    tagged = job.read(SOURCE_FILE).strip()
    template = load_prompt(cfg, "translate.txt")
    allowed = textcheck.load_allowed(cfg)
    target = textcheck.new_char_target(cfg)
    opts = schema.stage_opts(cfg, "translate")
    temps = schema.pass_temperatures(opts)
    prompt = build_prompt(template, tagged, allowed, target)
    meta_in = job.status().get("meta", {})
    expected = textcheck.extract_dnt(tagged) if meta_in.get("tagging") == "ok" else None
    tokens_in = estimate_tokens(prompt)
    job.log(f"translate: prompt is about {tokens_in:,} tokens; temperatures {temps}")

    for old in job.path.glob("04_translation_*.txt"):                 # left over from an earlier run
        old.unlink()

    llm.ensure_model_switched(cfg, opts["model"])
    limit = token_budget(cfg, "translate", tagged, factor=2.0, extra=800)
    candidates = []
    for n, temp in enumerate(temps, 1):
        entry = {"n": n, "temperature": temp, "file": None, "error": None}
        started = time.time()
        try:
            reply = llm.chat(cfg, prompt, model=opts["model"], temperature=temp, max_tokens=limit)
            body, notes = textcheck.split_reply(reply)
            body = re.sub(r"^```[\w-]*[ \t]*\n|\n```[ \t]*$", "", body).strip()
            if not body:
                raise RuntimeError("the reply was empty after removing the notes")
            name = f"04_translation_{n}.txt"
            job.write(name, body + "\n")
            entry.update(file=name, **summarise(textcheck.check(
                body + ("\nNOTES: " + notes if notes else ""), allowed, target, expected_dnt=expected)))
        except ConfigError:
            raise                                                   # a settings problem: no point trying again
        except (RuntimeError, OSError) as e:
            entry["error"] = str(e)
            job.log(f"translate: attempt {n} (temperature {temp}) failed: {e}")
        entry["seconds"] = round(time.time() - started, 1)
        candidates.append(entry)

    good = [c for c in candidates if c["file"]]
    job.write_json(META_FILE, {"temperatures": temps, "target_new_chars": target, "prompt_tokens": tokens_in,
                               "tagging": meta_in.get("tagging"), "candidates": candidates})
    if not good:
        raise RuntimeError("All " + str(len(candidates)) + " translation attempts failed. Last error: "
                           + str(candidates[-1]["error"]))

    parts = [f"#{c['n']} t={c['temperature']:g}: " + (f"{c['new_chars']} new" if c["file"] else "FAILED")
             for c in candidates]
    note = f"{len(good)} of {len(candidates)} translations; " + "; ".join(parts)
    flagged = [c for c in good if c["problems"]]
    if flagged:
        note += "; WARNING: problems in " + ", ".join(f"#{c['n']}" for c in flagged) + f" (see {META_FILE})"
    return note
