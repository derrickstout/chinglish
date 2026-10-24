"""Prompt files, output budgets and reply tidying, shared by every stage that calls a model.

    from core.prompts import load_prompt, token_budget, strip_wrapper

    prompt = load_prompt(cfg, "simplify.txt")            # text of prompts/simplify.txt
    limit = token_budget(cfg, "tag", story)               # max_tokens for a reply about as long as story
    text, notes = strip_wrapper(reply)                    # drop a code fence or "Here is the text:" line

load_prompt looks in the folder named by paths.prompts in config.json (default "prompts"). If that folder
is not there it also tries "Prompts" and "prompts", because the two are different folders on a case-sensitive
disk. Standard library only.
"""
import re
from pathlib import Path

from core import schema
from core.config import BASE, ConfigError, get

DEFAULT_DIR = "prompts"


def prompts_dir(cfg):
    """The folder holding the prompt files. Raises ConfigError (a plain message) if there is none."""
    configured = get(cfg, "paths.prompts", DEFAULT_DIR) or DEFAULT_DIR
    first = Path(configured).expanduser()
    first = first if first.is_absolute() else BASE / first
    candidates = [first] + [BASE / name for name in ("Prompts", "prompts")]
    for folder in candidates:
        if folder.is_dir():
            return folder
    raise ConfigError(f'The prompts folder was not found. Looked for "{first}". '
                      'Create it, or set "paths.prompts" in config.json.')


def load_prompt(cfg, filename):
    """The text of one prompt file, stripped. Raises ConfigError if it is missing or empty."""
    folder = prompts_dir(cfg)
    path = folder / filename
    if not path.is_file():
        raise ConfigError(f"The prompt file {filename} is not in {folder}.")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"Could not read {path}: {e}")
    if not text:
        raise ConfigError(f"The prompt file {path} is empty.")
    return text


def token_budget(cfg, stage, text, factor=1.0, extra=400):
    """max_tokens for a stage's reply, when the reply is roughly as long as `text` (times factor, plus some room).

    The stage's max_tokens_override is used exactly as written when it is set. Otherwise the limit is the
    model profile's max_tokens, raised if the text is long, so a long story is not cut off by the default.
    This only raises the ceiling on the reply; it does not change how much the model writes."""
    opts = schema.stage_opts(cfg, stage)
    if opts.get("max_tokens_override"):
        return opts["max_tokens_override"]
    configured = schema.profile(cfg, opts["model"])["max_tokens"]
    needed = int(len(text) / 2.5 * factor) + extra          # English runs about 4 characters a token; this is generous
    return max(configured, needed)


_FENCE = re.compile(r"^```[\w-]*[ \t]*\n|\n```[ \t]*$")
_PREAMBLE = re.compile(r"^(here('s| is| are)|sure|certainly|okay|ok|of course|below is)\b", re.I)


def strip_wrapper(reply):
    """Remove what models often put around an answer: a code fence around all of it, or a first line
    such as "Here is the simplified text:". Returns (text, notes); notes says what was removed."""
    notes = []
    text = reply.strip()
    unfenced = _FENCE.sub("", text).strip()
    if unfenced != text:
        notes.append("removed a code fence")
        text = unfenced
    lines = text.split("\n")
    if len(lines) > 1 and _PREAMBLE.match(lines[0].strip()) and lines[0].rstrip().endswith(":"):
        notes.append("removed an introduction line")
        text = "\n".join(lines[1:]).strip()
    return text, notes
