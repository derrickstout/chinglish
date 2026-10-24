#!/usr/bin/env python3
"""Chinese Study Buddy: run one stage of the pipeline, or all of it, from the terminal.

Start a new job from text (ingest or all only):
    python3 run.py all --text article.txt          # from a file
    python3 run.py all --text -                    # from standard input, same as piping:
    pbpaste | python3 run.py all
    python3 run.py all --clipboard                 # from the clipboard (macOS)

Work on a job that already exists:
    python3 run.py simplify --job 2026-10-17_1002_Serhiy_Beskrestnov
    python3 run.py translate --latest              # the newest job
    python3 run.py all --job ID --from translate   # run translate and everything after it again
    python3 run.py ingest --job ID --force         # run a finished stage again

Look at the settings:
    python3 run.py config                          # every model profile and stage, and any problem in config.json

Look at jobs:
    python3 run.py list                            # every job and what is next
    python3 run.py status --latest                 # one line per stage

Stages, in order: ingest, simplify, tag, translate, judge_translation, count, revert,
judge_comprehension, package, mail_out. A stage that is already done is skipped unless you add --force.
Stages that are not written yet say "not built yet" and `all` stops there.
"""
import argparse
import os
import re
import subprocess
import sys

import pipeline
from core import job as jobmod
from core import schema
from core.config import ConfigError, current_config
from pipeline import StageError
from stages import ingest


def read_clipboard():
    """The clipboard as text (macOS pbpaste)."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, check=True,
                                env={**os.environ, "LANG": "en_US.UTF-8"})
    except (OSError, subprocess.CalledProcessError):
        raise ConfigError("Could not read the clipboard (this needs the macOS pbpaste command). "
                          "Use --text FILE, or pipe the text in.")
    return result.stdout.decode("utf-8", "replace")


def read_stdin(stdin):
    data = stdin.buffer.read() if hasattr(stdin, "buffer") else stdin.read()
    return data.decode("utf-8-sig", "replace") if isinstance(data, bytes) else data


def read_text_file(path):
    try:
        with open(os.path.expanduser(path), encoding="utf-8-sig") as f:
            return f.read()
    except FileNotFoundError:
        raise ConfigError(f"File not found: {path}")
    except UnicodeDecodeError:
        raise ConfigError(f"{path} is not UTF-8 text. Save it as UTF-8 and try again.")
    except OSError as e:
        raise ConfigError(f"Could not read {path}: {e}")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="run.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0],
        epilog="Stages: " + ", ".join(pipeline.names()) + "\nSee the top of run.py for examples.")
    ap.add_argument("command", metavar="COMMAND", help="a stage name, all, status or list")
    ap.add_argument("--job", metavar="ID", help="an existing job folder name (see: run.py list)")
    ap.add_argument("--latest", action="store_true", help="the newest job")
    ap.add_argument("--text", metavar="FILE", help="start a new job from this text file (- = standard input)")
    ap.add_argument("--clipboard", action="store_true", help="start a new job from the clipboard (macOS)")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", help="with all: run this stage and every later one again")
    ap.add_argument("--force", action="store_true", help="run the stage even if it is already done")
    return ap


def _new_job_source(args, stdin, clipboard):
    """(text, label, slug) for a new job, or None when the arguments name no source."""
    if args.text and args.text != "-":
        stem = os.path.splitext(os.path.basename(args.text))[0]
        return read_text_file(args.text), f"file {args.text}", stem
    if args.text == "-":
        return read_stdin(stdin), "standard input", None
    if args.clipboard:
        return clipboard(), "the clipboard", None
    return None


def main(argv=None, cfg=None, stdin=None, clipboard=None, out=print):
    """Returns the exit code: 0 ok, 1 a problem with the work, 2 a problem with the arguments."""
    stdin = sys.stdin if stdin is None else stdin
    clipboard = clipboard or read_clipboard
    parser = build_parser()
    args = parser.parse_args(argv)

    stage_names = pipeline.names()
    if args.command not in stage_names + ["all", "status", "list", "config"]:
        parser.error(f"unknown command '{args.command}'. Choose a stage ({', '.join(stage_names)}), all, status, "
                     "list or config.")
    if sum(bool(x) for x in (args.job, args.latest, args.text, args.clipboard)) > 1:
        parser.error("choose one of --job, --latest, --text, --clipboard.")
    if args.from_stage and args.command != "all":
        parser.error("--from only goes with the all command.")

    try:
        cfg = cfg if cfg is not None else current_config()

        if args.command == "config":
            for line in schema.describe(cfg, os.environ):
                out(line)
            problems = schema.validate(cfg)
            for p in problems:
                out("PROBLEM: " + p)
            out("config.json has problems." if problems else "config.json looks fine.")
            return 1 if problems else 0
        if args.command not in ("list", "status"):
            schema.check(cfg)                                   # every problem at once, before any work starts

        if args.command == "list":
            jobs = jobmod.list_jobs(cfg)
            if not jobs:
                out("No jobs yet.")
            for j in jobs:
                nxt = pipeline.next_stage(j)
                out(f"  {j.id}   " + (f"next: {nxt}" if nxt else "complete"))
            return 0

        starts_new = args.command in ("ingest", "all")
        source = None
        if args.text or args.clipboard:
            if not starts_new:
                parser.error("--text and --clipboard only start a job with ingest or all. "
                             "Use --job ID or --latest for other stages.")
            source = _new_job_source(args, stdin, clipboard)
        elif not (args.job or args.latest):
            if args.command == "status":
                args.latest = True
            elif starts_new and not stdin.isatty():
                source = (read_stdin(stdin), "standard input", None)
            else:
                parser.error("say which job to use: --job ID, --latest, or give the text with "
                             "--text FILE, --clipboard, or by piping it in (ingest and all only).")

        if source is not None:
            text, label, slug = source
            if not text.strip():
                raise ConfigError(f"No text to work with: {label} is empty.")
            # A file is named after the file; pasted text after its first real words (not "View in browser").
            slug = _safe_slug(slug or ingest.clean(text, 0)[0])
            job = jobmod.new_job(cfg, text, slug=slug, source=label)
            out(f"Started job {job.id} from {label}.")
        elif args.latest:
            job = jobmod.latest_job(cfg)
            if job is None:
                raise ConfigError("There are no jobs yet. Start one with --text FILE.")
        else:
            job = jobmod.open_job(cfg, args.job)

        if args.command == "status":
            out(f"Job {job.id}")
            for line in pipeline.job_report(job):
                out(line)
            return 0

        code = 0
        if args.command == "all":
            pipeline.run_pipeline(cfg, job, start=args.from_stage, out=out)
        else:
            outcome = pipeline.run_stage(cfg, job, args.command, force=args.force, out=out)
            code = 1 if outcome == "not_built" else 0     # you asked for a stage that does not exist yet
        nxt = pipeline.next_stage(job)
        if nxt:
            out(f"Next: python3 run.py {nxt} --job {job.id}")
        if job.exists("09_result.txt"):
            out(f"Result: {job.file('09_result.txt')}")
        out(f"Job folder: {job.path}")
        return code

    except (ConfigError, StageError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _safe_slug(text):
    """A short folder-name label: the first few words, letters and digits only."""
    words = re.findall(r"[A-Za-z0-9]+", text or "")
    return "_".join(words[:6])[:40] or None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:                 # e.g. `python3 run.py list | head -1`
        sys.stderr.close()
        sys.exit(0)
