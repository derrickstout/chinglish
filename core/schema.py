"""The rules for config.json: every setting, its default, its type and range. Data and checks only.

    from core import schema

    schema.check(cfg)                       # raises ConfigError listing every problem, before any work starts
    opts = schema.stage_opts(cfg, "tag")    # that stage's settings, defaults filled in: opts["model"], ...
    prof = schema.profile(cfg, opts["model"])   # that model's settings: prof["url"], prof["max_tokens"], ...

config.json has one place for each kind of fact:
    "models"  named model profiles (any number, any names): url, model, max_tokens, api_key_env, ...
    "stages"  what each stage does: which profile it uses ("model"), how many passes, which temperature, ...
    "paths", "translation", "mail"   folders, the new-character goals, and email connection details
Steps that can be skipped have "enabled" (default true); "stages.mail_out.enabled" is false until you turn it on.
A stage never repeats a model's details; it names a profile. A profile that is not defined is an error.
Standard library only.
"""
import difflib

from core.config import ConfigError, get

REQUIRED = object()                     # a setting with no default: it must be in config.json

# A setting is (default, kind, low, high). kind: "str", "int", "float", "bool", "dict", "model" (a profile name),
# or a tuple of allowed strings. A float or int with default None means "not set: use the model profile's".

PROFILE = {
    "url": (REQUIRED, "str"),                  # the API's base address, without /chat/completions
    "model": (REQUIRED, "str"),                # the model's name as the server knows it
    "temperature": (0.3, "float", 0, 2),       # used when a stage does not set its own
    "max_tokens": (4096, "int", 1, None),      # the most the model may write in one reply (raised for long texts)
    "timeout": (900, "float", 1, None),        # seconds to wait for a reply
    "api_key_env": ("", "str"),                # NAME of the environment variable that holds the key; "" = no key
    "user_suffix": ("", "str"),                # text added to every prompt, e.g. "/no_think"
    "extra": ({}, "dict"),                     # any other fields to send with each request
    "server": ("", ("", "lm_studio")),         # "lm_studio" switches on the settings below
    "ttl": (600, "int", 0, None),              # lm_studio: seconds idle before the model is unloaded
    "reasoning_effort": ("", "str"),           # lm_studio: e.g. "none" switches thinking off
    "unload_on_switch": (True, "bool"),        # lm_studio: unload the old model when a stage uses another one
    "lms_path": ("", "str"),                   # lm_studio: where the 'lms' tool is, if it is not found
}

_ON = {"enabled": (True, "bool")}              # false = skip this step; the pipeline passes the text through
_MODEL = {"model": (REQUIRED, "model")}
_OVERRIDE = {"max_tokens_override": (None, "int", 1, None)}   # replaces the profile's max_tokens for this stage
_RETRY = {"retry_temp_step": (0.1, "float", 0, 2),     # each retry raises the temperature by this much
          "max_temperature": (1.0, "float", 0, 2)}     # ...but never above this
_TEXT_STAGE = {**_ON, **_MODEL, "temperature": (None, "float", 0, 2), "retries": (1, "int", 0, 5), **_RETRY,
               "on_fail": ("stop", ("continue", "stop")), **_OVERRIDE}
_PASSES = {**_MODEL, "passes": (3, "int", 1, 20), "temperature": (0.3, "float", 0, 2),
           "pass_temp_step": (0.3, "float", 0, 2), "max_temperature": (1.0, "float", 0, 2), **_OVERRIDE}
ALWAYS_ON = ("ingest", "translate", "count", "package")   # the pipeline cannot work without these
_JUDGE = {**_ON, **_MODEL, "temperature": (0.0, "float", 0, 2), "retries": (1, "int", 0, 5), **_RETRY, **_OVERRIDE}

STAGES = {                                     # in pipeline order
    "ingest": {"max_chars": (12000, "int", 0, None)},          # 0 = no cap
    "simplify": _TEXT_STAGE,                                    # on_fail: "stop", or "continue" with the text as it was
    "tag": {**_TEXT_STAGE, "retries": (2, "int", 0, 5), "on_fail": ("continue", ("continue", "stop"))},
    "translate": _PASSES,
    "judge_translation": _JUDGE,                                # off: the middle-temperature translation wins
    "count": {},
    "revert": {**_ON, **_PASSES},                               # off: the translation is kept as it is
    "judge_comprehension": _JUDGE,                              # off: the middle valid reversion wins
    "package": {},
    "mail_out": {"enabled": (False, "bool"),                    # email the result; off unless you turn it on
                 "dry_run": (False, "bool")},                   # true: build and save the email, do not send it
}

OTHER = {
    "paths": {k: ("", "str") for k in ("jobs", "prompts", "data", "samples")},
    "translation": {"target_new_chars": (50, "int", 0, None), "revert_tolerance": (10, "int", 0, None)},
    "mail": None,                              # checked by the mail stage itself
}

REMOVED = {                                    # old settings and where they went
    "local_llm": 'was replaced by a profile in "models" (and its stages name it in "stages.<stage>.model")',
    "cloud_llm": 'was replaced by a profile in "models" (and its stages name it in "stages.<stage>.model")',
    "ingest": 'moved to "stages.ingest" (stages.ingest.max_chars)',
}
REMOVED_IN_MAIL = {
    "enabled": 'moved to "stages.mail_out.enabled" (the mail section now holds only connection details)',
}
REMOVED_IN_TRANSLATION = {
    "translate_temperatures": 'was replaced by "stages.translate": passes, temperature, pass_temp_step',
    "revert_temperatures": 'was replaced by "stages.revert": passes, temperature, pass_temp_step',
}


def _guess(name, options):
    close = difflib.get_close_matches(str(name), list(options), n=1)
    return f' (did you mean "{close[0]}"?)' if close else ""


def _coerce(value, spec):
    """The value in its proper type, or raises ValueError with what was expected."""
    kind, low, high = spec[1], (spec[2] if len(spec) > 2 else None), (spec[3] if len(spec) > 3 else None)
    if value is None and spec[0] is None:
        return None
    if isinstance(kind, tuple):
        if value not in kind:
            raise ValueError("must be one of " + ", ".join(f'"{k}"' for k in kind))
        return value
    if kind in ("str", "model"):
        if not isinstance(value, str):
            raise ValueError("must be text, in quotes")
        if kind == "model" and not value.strip():
            raise ValueError("must name a model profile")
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("must be true or false, without quotes")
        return value
    if kind == "dict":
        if not isinstance(value, dict):
            raise ValueError("must be an object like {}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number, without quotes")
    if kind == "int":
        if value != int(value):
            raise ValueError("must be a whole number")
        value = int(value)
    else:
        value = float(value)
    if low is not None and value < low:
        raise ValueError(f"must be at least {low}")
    if high is not None and value > high:
        raise ValueError(f"must be at most {high}")
    return value


def _section(raw, spec, path, problems):
    """One settings block with defaults filled in. Problems are added to the list, never raised."""
    out = {}
    if not isinstance(raw, dict):
        problems.append(f'"{path}" must be an object like {{}}.')
        raw = {}
    for key in raw:
        if key not in spec:
            problems.append(f'"{path}.{key}" is not a setting{_guess(key, spec)}. '
                            f"Settings here: {', '.join(spec) or '(none)'}.")
    for key, s in spec.items():
        if key in raw and not (s[0] is REQUIRED and raw[key] in ("", None)):
            try:
                out[key] = _coerce(raw[key], s)
            except ValueError as e:
                problems.append(f'"{path}.{key}" {e}.')
        elif s[0] is REQUIRED:
            problems.append(f'Set "{path}.{key}" in config.json.')
        else:
            out[key] = s[0]
    return out


# ---------------------------------------------------------------------------- the two lookups stages use

def profile(cfg, name):
    """One model profile, defaults filled in. Raises ConfigError if it is missing or wrong."""
    models = get(cfg, "models")
    if not isinstance(models, dict) or not models:
        raise ConfigError('Set up "models" in config.json: one entry per model, for example "local1".')
    if name not in models:
        raise ConfigError(f'The model "{name}" is not defined in "models". Defined: {", ".join(models)}.')
    problems = []
    out = _section(models[name], PROFILE, f"models.{name}", problems)
    if problems:
        raise ConfigError(" ".join(problems))
    return out


def stage_opts(cfg, name):
    """One stage's settings, defaults filled in. Raises ConfigError if they are wrong, or if the stage's
    "model" is not a defined profile."""
    if name not in STAGES:
        raise ConfigError(f'Unknown stage "{name}". Stages: {", ".join(STAGES)}.')
    raw = dict(get(cfg, f"stages.{name}", {}) or {})
    problems = []
    if name in ALWAYS_ON and "enabled" in raw:
        raw.pop("enabled")
        problems.append(f'"stages.{name}.enabled" cannot be used: the pipeline cannot work without the "{name}" step. '
                        f'Steps you can turn off: {", ".join(n for n in STAGES if "enabled" in STAGES[n])}.')
    out = _section(raw, STAGES[name], f"stages.{name}", problems)
    if "model" in out:
        models = get(cfg, "models") or {}
        if out["model"] not in models:
            problems.append(f'"stages.{name}.model" is "{out["model"]}", which is not defined in "models" '
                            f'(defined: {", ".join(models) or "none"}).')
    if problems:
        raise ConfigError(" ".join(problems))
    return out


def is_enabled(cfg, name):
    """False only when the stage can be turned off and stages.<name>.enabled says so."""
    return True if name in ALWAYS_ON or name not in STAGES else stage_opts(cfg, name)["enabled"]


def _stepped(base, step, cap, n):
    """base plus n steps, never above cap (a base that is already above the cap is left as it is)."""
    return round(min(base + n * step, max(cap, base)), 3)


def pass_temperature(opts, n):
    """The temperature for pass n (0 is the first): the start plus n steps, never above max_temperature."""
    return _stepped(opts["temperature"], opts["pass_temp_step"], opts["max_temperature"], n)


def attempt_temperature(cfg, opts, n):
    """The temperature for attempt n (0 is the first, 1 the first retry). None means "the model profile's own"
    and is only possible for the first attempt of a stage that sets no temperature."""
    base = opts["temperature"]
    if base is None:
        if n == 0:
            return None
        base = profile(cfg, opts["model"])["temperature"]
    if n == 0:
        return base
    return round(min(base + n * opts["retry_temp_step"], opts["max_temperature"]), 3)


def pass_temperatures(opts):
    return [pass_temperature(opts, n) for n in range(opts["passes"])]


# ---------------------------------------------------------------------------- checking a whole file

def validate(cfg):
    """Every problem in the settings, as a list of plain sentences (empty when all is well)."""
    problems = []
    if not isinstance(cfg, dict):
        return ["config.json must hold a JSON object."]
    allowed = ["models", "stages", *OTHER]
    for key in cfg:
        if key in REMOVED:
            problems.append(f'"{key}" {REMOVED[key]}.')
        elif key not in allowed:
            problems.append(f'"{key}" is not a section of config.json{_guess(key, allowed)}. '
                            f"Sections: {', '.join(allowed)}.")

    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        problems.append('Set up "models" in config.json: one entry per model, for example "local1".')
        models = {}
    for name, raw in models.items():
        _section(raw, PROFILE, f"models.{name}", problems)

    raw_stages = cfg.get("stages", {})
    if not isinstance(raw_stages, dict):
        problems.append('"stages" must be an object like {}.')
        raw_stages = {}
    for name in raw_stages:
        if name not in STAGES:
            problems.append(f'"stages.{name}" is not a stage{_guess(name, STAGES)}. Stages: {", ".join(STAGES)}.')
    for name in STAGES:
        if name in raw_stages or any(s[0] is REQUIRED for s in STAGES[name].values()):
            try:
                stage_opts(cfg, name)
            except ConfigError as e:
                problems.append(str(e))

    if isinstance(cfg.get("mail"), dict):
        for key in REMOVED_IN_MAIL:
            if key in cfg["mail"]:
                problems.append(f'"mail.{key}" {REMOVED_IN_MAIL[key]}.')
    for section, spec in OTHER.items():
        if spec is None or section not in cfg:
            continue
        raw = cfg[section]
        if section == "translation" and isinstance(raw, dict):
            for key in REMOVED_IN_TRANSLATION:
                if key in raw:
                    problems.append(f'"translation.{key}" {REMOVED_IN_TRANSLATION[key]}.')
            raw = {k: v for k, v in raw.items() if k not in REMOVED_IN_TRANSLATION}
        _section(raw, spec, section, problems)
    return problems


def check(cfg):
    """Raise ConfigError naming every problem, or return quietly."""
    problems = validate(cfg)
    if problems:
        raise ConfigError("config.json has problems:\n  - " + "\n  - ".join(problems))


def describe(cfg, environ):
    """Lines that show what a run would do: each model profile, and each stage with its model and numbers."""
    lines = ["Models:"]
    models = cfg.get("models") if isinstance(cfg.get("models"), dict) else {}
    used = {}
    for name in STAGES:
        try:
            opts = stage_opts(cfg, name)
        except ConfigError:
            continue
        if "model" in opts and opts.get("enabled", True):
            used.setdefault(opts["model"], []).append(name)
    for name in models:
        try:
            p = profile(cfg, name)
        except ConfigError:
            lines.append(f"  {name:<8} (has problems)")
            continue
        key = ""
        if p["api_key_env"]:
            key = f"  key {p['api_key_env']}: " + ("set" if environ.get(p["api_key_env"]) else "NOT SET")
        used_by = f"  used by {', '.join(used[name])}" if name in used else "  (not used)"
        lines.append(f"  {name:<8} {p['server'] or 'api':<10} {p['model']}  max_tokens {p['max_tokens']}{key}{used_by}")
    lines.append("Stages:")
    for name in STAGES:
        try:
            o = stage_opts(cfg, name)
        except ConfigError:
            lines.append(f"  {name:<20} (has problems)")
            continue
        if o.get("enabled") is False:
            lines.append(f"  {name:<20} OFF (stages.{name}.enabled is false)")
            continue
        bits = [f"{k} {v}" for k, v in o.items() if v is not None and k not in ("model", "enabled")]
        lines.append((f"  {name:<20} " + (f"model {o['model']}  " if "model" in o else "") + ", ".join(bits)).rstrip())
    return lines
