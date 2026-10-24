"""The one place that talks to a model: any OpenAI-compatible server (LM Studio, a cloud API, ...).

    from core.llm import chat
    reply = chat(cfg, prompt, text, model="local1")            # prompt = system message, text = user message
    reply = chat(cfg, messages=[...], model="local1", temperature=0.6)   # a whole conversation

"model" is the name of a profile in the "models" section of config.json; every detail (url, model name,
max_tokens, key variable, ...) comes from that profile and nowhere else. Stages get the name from their own
"stages.<stage>.model" setting. A key is read from the environment variable named in the profile's
api_key_env, never from the file.

Check a profile from Terminal:
    python3 -m core.llm --test              # lists the profiles you have defined
    python3 -m core.llm --test cloud1       # lists that server's models, then asks one question

Standard library only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from core import schema
from core.config import ConfigError, current_config

LOADED_MODEL = {"name": None}          # the local model this program last asked LM Studio to use


# ---------- transport (the two functions tests replace with a fake) ----------

def _post(url, body, headers, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, headers, timeout):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------- settings for one profile ----------

def _connection(cfg, name, need_model=True):
    """(profile, base url, request headers) for a model profile. Raises ConfigError if something is not set.
    need_model=False is for listing a server's models before one has been chosen."""
    if not name:
        raise ConfigError("No model was named. Each stage needs a profile: set stages.<stage>.model in config.json.")
    p = schema.profile(cfg, name)
    headers = {"Content-Type": "application/json"}
    env = p["api_key_env"]
    if env:
        key = os.environ.get(env, "")
        if not key:
            raise ConfigError(f'The API key for the model "{name}" is not set. In Terminal, run: export {env}="your key"')
        headers["Authorization"] = "Bearer " + key
    return p, p["url"].rstrip("/"), headers


def _request_body(p, messages, temperature, max_tokens):
    """The JSON sent to /chat/completions."""
    body = {
        "model": p["model"],
        "temperature": p["temperature"] if temperature is None else temperature,
        "max_tokens": max_tokens or p["max_tokens"],
        "messages": messages,
    }
    if p["server"] == "lm_studio":
        body["ttl"] = p["ttl"]                                    # LM Studio: unload after this many idle seconds
        if p["reasoning_effort"]:                                 # e.g. "none" switches thinking off
            body["reasoning_effort"] = p["reasoning_effort"]
    extra = dict(p["extra"])
    if temperature is not None:
        extra.pop("temperature", None)                            # an explicit temperature wins over "extra"
    body.update(extra)
    return body


# ---------- the chat call ----------

def chat(cfg, prompt=None, text=None, messages=None, temperature=None, max_tokens=None, model=None):
    """Send a prompt to the named model profile and return its reply as plain text.

    With text, prompt is the system message and text is the user message.
    With only prompt, it is sent as a single user message.
    With messages (a list of {"role", "content"}), that whole conversation is sent instead.
    The profile's user_suffix goes on the last user message.
    temperature and max_tokens, when given, replace the profile's."""
    p, url, headers = _connection(cfg, model)
    suffix = p["user_suffix"]
    if messages is not None:
        messages = [dict(m) for m in messages]
        if suffix and messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = (messages[-1]["content"] + "\n" + suffix).strip()
    elif text is None:
        messages = [{"role": "user", "content": (prompt + "\n" + suffix).strip()}]
    else:
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": (text + "\n" + suffix).strip()}]
    body = _request_body(p, messages, temperature, max_tokens)

    started = time.time()
    try:
        r = _post(url + "/chat/completions", body, headers, p["timeout"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'The model "{model}" ({url}) returned an error: ' + e.read().decode("utf-8", "replace")[:300])
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f'Could not reach the model "{model}" at {url}: {e}')
    used = (r.get("usage") or {}).get("completion_tokens")
    if used:                                                      # handy for comparing models
        secs = max(time.time() - started, 0.1)
        print(f"  model wrote {used:,} tokens in {secs:.0f}s ({used / secs:.1f} tokens/s)")
    choice = r["choices"][0]
    reply = re.sub(r"<think>.*?</think>", "", choice["message"].get("content") or "", flags=re.S).strip()
    if not reply:
        finish = choice.get("finish_reason")
        if finish == "length":
            raise RuntimeError(
                f"The model hit the max_tokens limit ({body['max_tokens']}) before writing an answer. "
                "This is separate from context length. A thinking model can use thousands of tokens "
                f'reasoning first: raise models.{model}.max_tokens (or the stage\'s max_tokens_override), '
                "switch off thinking, or use a non-thinking model.")
        raise RuntimeError(f"The model returned no text (finish reason: {finish}).")
    return reply


def list_models(cfg, model, timeout=10):
    """The model names the server behind a profile lists."""
    _p, url, headers = _connection(cfg, model, need_model=False)
    r = _get(url + "/models", headers, timeout)
    return [m["id"] for m in r.get("data", [])]


# ---------- LM Studio model switching (profiles with server "lm_studio") ----------

def find_lms(cfg, name):
    """LM Studio's command-line tool, or None. Set models.<profile>.lms_path in config.json if it isn't found."""
    for candidate in (schema.profile(cfg, name)["lms_path"], shutil.which("lms"), Path.home() / ".lmstudio" / "bin" / "lms"):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def ensure_model_switched(cfg, name):
    """If this LM Studio profile's model differs from the model used last (or this is the first call since
    the program started), unload everything from LM Studio so the new model has the memory to load.
    Does nothing for a profile that is not an lm_studio server. Set the profile's unload_on_switch to false
    to skip the unloading. This is the only step that needs the 'lms' tool; model requests do not.
    Call it once per stage, before the stage's first model call."""
    p = schema.profile(cfg, name)
    wanted = p["model"]
    if p["server"] != "lm_studio" or LOADED_MODEL["name"] == wanted:
        return
    if not p["unload_on_switch"]:
        print(f"  model: {wanted} (unload_on_switch is off, so LM Studio is not asked to unload anything)")
        LOADED_MODEL["name"] = wanted
        return
    previous = LOADED_MODEL["name"] or "whatever was loaded before"
    lms = find_lms(cfg, name)
    if not lms:
        print("  model changed but the 'lms' tool was not found, so LM Studio's old model was not "
              f'unloaded. Set "models.{name}.lms_path" in config.json, or unload it in LM Studio yourself.')
    else:
        print(f"  switching model: {previous} -> {wanted}; unloading LM Studio's loaded models first")
        try:
            r = subprocess.run([lms, "unload", "--all"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print(f"  lms unload failed: {(r.stderr or r.stdout).strip()[:200]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"  lms unload failed: {e}")
    LOADED_MODEL["name"] = wanted


# ---------- command line: connection test ----------

def test_connection(cfg, name):
    try:
        p, url, headers = _connection(cfg, name)
        models = list_models(cfg, name)
    except ConfigError as e:
        sys.exit(str(e))
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f'Could not reach the model "{name}" at {_url_of(cfg, name)}: {e}'
                 + ("\nIs the LM Studio server running?" if p["server"] == "lm_studio" else ""))
    print(f'Models the server behind "{name}" lists:')
    for m in models:
        print("  " + m)
    print(f"\nAsking {p['model']} a test question (the first call can be slow while a local model loads)...")
    question = ("Reply with the single word: ready\n" + p["user_suffix"]).strip()
    body = _request_body(p, [{"role": "user", "content": question}], 0, 300)
    try:
        r = _post(url + "/chat/completions", body, headers, 300)
        choice = r["choices"][0]
        msg = choice["message"]
        reply = (msg.get("content") or "").strip()
        print("Model replied:", reply or "(nothing)")
        print("Finish reason:", choice.get("finish_reason"))
        print("Tokens generated:", (r.get("usage") or {}).get("completion_tokens", "unknown"),
              "(a one-word answer with thinking off is usually under 10)")
        if msg.get("reasoning_content"):
            print(f"The model also produced {len(msg['reasoning_content'])} characters of hidden reasoning.")
        if not reply:
            print("The reply was empty. This is what the server sent back:")
            print(json.dumps(msg, ensure_ascii=False, indent=1)[:600])
    except urllib.error.HTTPError as e:
        sys.exit("The request failed: " + e.read().decode("utf-8", "replace")[:300])
    except (urllib.error.URLError, OSError, KeyError) as e:
        sys.exit(f"The request failed: {e}")


def _url_of(cfg, name):
    try:
        return schema.profile(cfg, name)["url"]
    except ConfigError:
        return "?"


def main():
    ap = argparse.ArgumentParser(description="Test the connection to a model profile from config.json.")
    ap.add_argument("--test", nargs="?", const="", metavar="PROFILE",
                    help="list the server's models and ask one question; with no name, list your profiles")
    args = ap.parse_args()
    if args.test is None:
        ap.print_help()
        return
    try:
        cfg = current_config()
    except ConfigError as e:
        sys.exit(str(e))
    if not args.test:
        names = list((cfg.get("models") or {}))
        sys.exit("Your model profiles: " + (", ".join(names) or "none") + "\nTest one with: python3 -m core.llm --test NAME")
    test_connection(cfg, args.test)


if __name__ == "__main__":
    main()
