# Chinese Study Buddy

A terminal pipeline that turns an English article into a Traditional Chinese version written for a learner. It prefers characters from your own known-character list, protects names and acronyms, and puts English back where the Chinese would use too many new characters.

Every stage is a separate command, reads and writes plain files, and can be run, rerun or skipped on its own. Python standard library only; no packages to install.

## How it works

```
text (file, clipboard or pipe)
  -> ingest             clean the text
  -> simplify           simpler English                              (optional)
  -> tag                wrap names, acronyms and quotes in <dnt>     (optional)
  -> translate          3 passes into Traditional Chinese, each a little warmer
  -> judge_translation  a model picks the best of the three          (optional)
  -> count              count characters outside your known list
  -> revert             only if over target: English put back in <en> tags, 3 passes   (optional)
  -> judge_comprehension  a model picks the most readable reversion  (optional)
  -> package            write the result file
  -> mail_out           email the result                             (optional, off by default)
```

Each article gets a job folder, `jobs/<date_time_slug>/`, holding one file per stage (`00_source.txt` ... `10_sent.json`), `job.json` (what has finished) and `log.txt`. The file you read is `09_result.txt`: the translation, the simplified English, the original article and notes. Warnings come first.

`new characters` means Chinese characters outside the `mastered` and `learning` lists in `data/allowed_chars.json`. `translation.target_new_chars` is a soft target: accuracy comes first, and the revert step runs only when the translation is more than `revert_tolerance` over it.

## Requirements

- Python 3.14 (developed on macOS; the clipboard option uses `pbpaste`).
- A model for the local steps: [LM Studio](https://lmstudio.ai) with its server running (default `http://localhost:1234/v1`). Any OpenAI-compatible server works.
- A model for the judging steps: the shipped config uses Gemini through its OpenAI-compatible endpoint, with a key from Google AI Studio. A local model also works; point those stages at a local profile.

## Setup

1. Put the project in a folder and open Terminal there.
2. Add your own prompts to `prompts/`: `simplify.txt` (rewrite in simple English) and `dnt.txt` (the tagging prompt; the story is appended after it). The other prompts ship with the project.
3. Put your character lists in `data/allowed_chars.json`:
   ```json
   {"mastered": "北本不出...", "learning": "愛安..."}
   ```
   Each list is one string of characters.
4. Set your keys in Terminal, once per session (they are never stored in config.json):
   ```
   export GEMINI_API_KEY="your key"
   export MAIL_PASSWORD="your app password"     # only if you turn mail on
   ```
5. Edit `config.json` (next section), then check it:
   ```
   python3 run.py config
   ```
6. Optional connection checks:
   ```
   python3 -m core.llm --test local1
   python3 -m core.llm --test cloud1
   ```

## First run

```
python3 run.py all --text samples/some_article.txt
```

The command prints one line per stage and, when finished, the path of `09_result.txt`. A finished stage is skipped if you run the command again.

## Commands

| Goal | Command |
| --- | --- |
| Check config.json | `python3 run.py config` |
| Run everything on a file | `python3 run.py all --text article.txt` |
| Run everything on the clipboard | `python3 run.py all --clipboard` |
| Run everything on piped text | `pbpaste \| python3 run.py all` |
| List jobs and what is next | `python3 run.py list` |
| One line per stage for a job | `python3 run.py status --latest` |
| Run one stage | `python3 run.py translate --latest` or `--job ID` |
| Redo a finished stage | `python3 run.py ingest --job ID --force` |
| Redo a stage and everything after it | `python3 run.py all --job ID --from translate` |
| Send a test email | `python3 -m stages.mail_out --test` |
| Run the tests | `python3 -m unittest discover tests` |

Stage names, in order: `ingest`, `simplify`, `tag`, `translate`, `judge_translation`, `count`, `revert`, `judge_comprehension`, `package`, `mail_out`.

## Configuration

`config.json` is the only file you edit for settings, and `core/schema.py` defines the rules that check it. `python3 run.py config` lists every problem at once, with the fix.

| Section | What it holds |
| --- | --- |
| `models` | Any number of named profiles (`local1`, `local2`, `cloud1`, `cloud2`, or your own): `url`, `model`, `temperature`, `max_tokens`, `timeout`, `api_key_env`, plus LM Studio options (`ttl`, `reasoning_effort`, `unload_on_switch`, `lms_path`). All connection details live here and nowhere else. |
| `stages` | One entry per step. It names a profile with `"model": "local1"` and sets that step's `retries`, `passes`, temperatures, `on_fail` and `enabled`. |
| `paths` | The `jobs`, `prompts` and `data` folders. |
| `translation` | `target_new_chars` (default 50) and `revert_tolerance` (default 10). |
| `mail` | SMTP connection details only. |

Useful stage settings:

- **Different models per step.** Give each stage a different profile, for example `"tag": {"model": "local2"}`. Add as many profiles as you like.
- **Skipping.** `"enabled": false` turns off `simplify`, `tag`, `judge_translation`, `revert`, `judge_comprehension` or `mail_out`. The text is passed through unchanged, and the result notes list what was off. `ingest`, `translate`, `count` and `package` cannot be turned off.
- **Retries.** `tag` (2), `simplify` (1) and both judges (1) retry with a slightly warmer temperature, set by `retry_temp_step` and capped by `max_temperature`. A rejected tag reply is retried with the reason added from `prompts/dnt_retry.txt`.
- **When retries run out.** `on_fail` decides: `tag` defaults to `continue` (go on with no tags, and warn), `simplify` to `stop`. A server that never answers always fails the stage.
- **Passes.** `translate` and `revert` use `passes`, `temperature` and `pass_temp_step`.
- **Mail.** `stages.mail_out.enabled` is false by default. `dry_run: true` builds `10_email.eml` without sending it.
- **LM Studio memory.** `unload_on_switch: true` on a local profile runs `lms unload --all` when the model changes, which helps when two models do not fit in memory at once. It is not needed to use different models.

## Layout

```
config.json        settings
run.py             the command you type
pipeline.py        stage order, skipping and resuming
core/              config, schema, llm, job, prompts, tags, textcheck
stages/            ingest simplify tag translate judge count revert package mail_out
prompts/           the prompt text for each model step
data/              allowed_chars.json
samples/           articles to try
tests/             unit tests (a scripted fake model; no server needed) and live_check_*.py
jobs/              one folder per article (created on first run)
```

## Tests

```
python3 -m unittest discover tests
```

The tests use a scripted fake model, so they need no LM Studio and no API key. `tests/live_check_*.py` are optional scripts that talk to your real models.

## Status

Working: every stage above, with retries, per-stage model profiles, skipping and config validation.

Not built yet: reading incoming email (a trigger that starts a job from a message and replies to its sender), and a web interface. The config layout is designed so a web UI can later read and write the same settings.

Not yet checked: a full run against real LM Studio and Gemini models with your prompts.
