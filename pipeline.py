"""The pipeline: which stages exist, in what order, and how one stage is run, skipped or resumed.

This file holds no translation logic. It only knows each stage's name, module, the files it needs
and the files it makes, then:

    run_stage(cfg, job, "simplify")        run one stage (skipped if it is already done)
    run_pipeline(cfg, job)                 run every stage in order, from the start or from a stage
    job_report(job)                        one line per stage: done / failed / skipped / pending

A stage that has not been written yet is reported as "not built yet" and the pipeline stops there,
so the pipeline can be run while it is still being built.

Rules:
  * A finished stage is skipped on a rerun, as long as its output files are still there.
  * Running a stage again (force, or --from) clears the finished marks of every later stage, so a
    stale result is never reused.
  * A stage that stages.<name>.enabled turns off is not run: simplify, tag and the judges pass their text
    through (see Stage.skip), revert and mail_out are marked "skipped".
  * A conditional stage (reversion) whose condition is false is marked "skipped", and its old
    output files are removed so a later stage cannot pick up a stale one.
"""
import fnmatch
import importlib
import time
import traceback
from dataclasses import dataclass

from core import schema
from core.config import ConfigError


class StageError(RuntimeError):
    """A stage could not run. The message is worded for the person at the terminal."""


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    func: str = "run"
    needs: tuple = ()       # files that must exist before the stage starts
    makes: tuple = ()       # files that must exist after it finishes
    when: object = None     # optional function(job) -> bool; the stage is skipped when it is False
    cleans: tuple = ()      # glob patterns removed from the job folder when the stage is skipped
    skip: str = None        # function(cfg, job) -> note that stands in for the stage when it is turned off
                            # (it writes the files the next stage needs); None: just skip it
    follows: str = None     # a stage that this one works on: when that one is turned off, so is this one


def reversion_needed(job):
    """True when count.py decided the winning translation is far enough over the target, False when it
    decided not. None when count has not run yet, so nobody can tell (run_stage then asks for 06_count.json)."""
    try:
        return bool(job.read_json("06_count.json").get("revert_needed"))
    except (FileNotFoundError, ValueError):
        return None


STAGES = [
    Stage("ingest", "stages.ingest", needs=("00_source.txt",), makes=("01_clean.txt",)),
    Stage("simplify", "stages.simplify", needs=("01_clean.txt",), makes=("02_simplified.txt",), skip="skip"),
    Stage("tag", "stages.tag", needs=("02_simplified.txt",), makes=("03_tagged.txt",), skip="skip"),
    Stage("translate", "stages.translate", needs=("03_tagged.txt",), makes=("04_meta.json",)),
    Stage("judge_translation", "stages.judge", "run_translation", needs=("04_meta.json",), makes=("05_winner.txt",),
          skip="skip_translation"),
    Stage("count", "stages.count", needs=("05_winner.txt",), makes=("06_count.json",)),
    Stage("revert", "stages.revert", needs=("06_count.json",), makes=("07_meta.json",),
          when=reversion_needed, cleans=("07_*",)),
    Stage("judge_comprehension", "stages.judge", "run_comprehension", needs=("07_meta.json",),
          makes=("08_winner.txt",), when=reversion_needed, cleans=("08_*",),
          skip="skip_comprehension", follows="revert"),
    Stage("package", "stages.package", needs=("01_clean.txt", "05_winner.txt"), makes=("09_result.txt", "09_result.json")),
    Stage("mail_out", "stages.mail_out", needs=("09_result.json",), makes=("10_sent.json",)),
]


def names():
    return [s.name for s in STAGES]


def get_stage(name):
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ConfigError(f"Unknown stage '{name}'. Stages, in order: {', '.join(names())}.")


def load_runner(stage):
    """The stage's function, or None if its module has not been written yet."""
    try:
        module = importlib.import_module(stage.module)
    except ModuleNotFoundError as e:
        if e.name and (e.name == stage.module or stage.module.startswith(e.name + ".")):
            return None
        raise
    fn = getattr(module, stage.func, None)
    if fn is None:
        raise StageError(f"{stage.module} has no function {stage.func}().")
    return fn


# ----------------------------------------------------------------------- job status helpers

def _is_complete(job, stage):
    return job.stage_done(stage.name) and all(job.exists(f) for f in stage.makes)


def _set_status(job, name, entry):
    data = job.status()
    data.setdefault("stages", {})[name] = entry
    job.write_json("job.json", data)


def _invalidate_after(job, name):
    """Forget that any stage after `name` has finished, because its inputs are about to change."""
    order = names()
    data = job.status()
    stages = data.get("stages", {})
    removed = [s for s in order[order.index(name) + 1:] if s in stages]
    if not removed:
        return
    for s in removed:
        del stages[s]
        _remove_files(job, get_stage(s).cleans)          # their output files too, so nothing stale is picked up
    done = [s for s in order if stages.get(s, {}).get("status") == "done"]
    if done:
        data["last_finished"] = done[-1]
    else:
        data.pop("last_finished", None)
    job.write_json("job.json", data)


def _remove_files(job, patterns):
    removed = []
    for path in sorted(job.path.iterdir()):
        if any(fnmatch.fnmatch(path.name, p) for p in patterns):
            path.unlink()
            removed.append(path.name)
    return removed


def next_stage(job):
    """The first stage that has not finished (and was not skipped), or None when all have."""
    stages = job.status().get("stages", {})
    for name in names():
        if stages.get(name, {}).get("status") not in ("done", "skipped"):
            return name
    return None


def job_report(job):
    """One line per stage, for `run.py status`."""
    stages = job.status().get("stages", {})
    lines = []
    for name in names():
        entry = stages.get(name)
        if not entry:
            lines.append(f"  {name:<20} pending")
            continue
        status = entry.get("status", "?")
        detail = ""
        if status == "done" and "seconds" in entry:
            detail = f" ({entry['seconds']}s)"
        note = entry.get("note") or entry.get("error") or ""
        lines.append(f"  {name:<20} {status}{detail}" + (f"  {note}" if note else ""))
    return lines


# ----------------------------------------------------------------------- running

def _describe(exc):
    text = str(exc).strip()
    return text if isinstance(exc, (RuntimeError, ConfigError, FileNotFoundError, ValueError)) and text \
        else f"{type(exc).__name__}: {text}"


def _turned_off(cfg, job, stage):
    """True when config.json turns this stage off, or turns off the stage it follows. A stage whose condition
    is plainly false is "not needed" instead, whatever the switch says."""
    if stage.when is not None and stage.when(job) is False:
        return False
    return not schema.is_enabled(cfg, stage.name) or (stage.follows is not None and not schema.is_enabled(cfg, stage.follows))


def _run_turned_off(cfg, job, stage, out):
    """A stage that is switched off: pass its input through if it has a stand-in, and mark it skipped."""
    name = stage.name
    note = "turned off in config.json"
    stand_in = stage.skip and (stage.follows is None or schema.is_enabled(cfg, stage.follows))
    if stand_in:
        missing = [f for f in stage.needs if not job.exists(f)]
        if missing:
            producers = [s.name for s in STAGES if any(f in s.makes for f in missing)]
            hint = f" Run: python3 run.py {producers[0]} --job {job.id}" if producers else ""
            raise StageError(f"{name} needs {', '.join(missing)}, which is not in the job folder.{hint}")
        _invalidate_after(job, name)
        fn = getattr(importlib.import_module(stage.module), stage.skip)
        try:
            note += ": " + (fn(cfg, job) or "")
        except Exception as e:
            job.log(traceback.format_exc())
            job.mark_failed(name, _describe(e))
            raise StageError(f"{name} failed: {_describe(e)} (details: {job.file('log.txt')})") from e
    else:
        _invalidate_after(job, name)
        _remove_files(job, stage.cleans)
    _set_status(job, name, {"status": "skipped", "note": note.rstrip(": ")})
    job.log(f"{name}: {note}")
    out(f"  {name}: {note}")
    return "turned_off"


def run_stage(cfg, job, name, force=False, out=print):
    """Run one stage. Returns "done", "skipped" (already done), "turned_off", "not_needed" or "not_built";
    raises StageError."""
    stage = get_stage(name)

    if not force and _is_complete(job, stage):
        out(f"  {name}: already done (add --force to run it again)")
        return "skipped"

    if _turned_off(cfg, job, stage):
        entry = job.status().get("stages", {}).get(name, {})
        if not force and entry.get("status") == "skipped" and entry.get("note", "").startswith("turned off"):
            out(f"  {name}: turned off in config.json (stages.{name}.enabled)")
            return "turned_off"
        return _run_turned_off(cfg, job, stage, out)

    runner = load_runner(stage)
    if runner is None:
        out(f"  {name}: not built yet")
        return "not_built"

    if stage.when is not None and stage.when(job) is False:      # None means "cannot tell yet": check the inputs
        removed = _remove_files(job, stage.cleans)
        _set_status(job, name, {"status": "skipped", "note": "not needed"})
        out(f"  {name}: not needed" + (f" (removed old {', '.join(removed)})" if removed else ""))
        job.log(f"{name}: not needed")
        return "not_needed"

    missing = [f for f in stage.needs if not job.exists(f)]
    if missing:
        producers = [s.name for s in STAGES if any(f in s.makes for f in missing)]
        hint = (f" Run: python3 run.py {producers[0]} --job {job.id}" if producers else "")
        raise StageError(f"{name} needs {', '.join(missing)}, which is not in the job folder.{hint}")

    _invalidate_after(job, name)
    started = time.time()
    job.log(f"{name}: started")
    try:
        note = runner(cfg, job) or ""
    except StageError:
        raise
    except Exception as e:
        job.log(traceback.format_exc())
        job.mark_failed(name, _describe(e))
        raise StageError(f"{name} failed: {_describe(e)} (details: {job.file('log.txt')})") from e

    absent = [f for f in stage.makes if not job.exists(f)]
    if absent:
        message = f"{name} finished but did not write {', '.join(absent)}."
        job.mark_failed(name, message)
        raise StageError(message)

    seconds = time.time() - started
    job.mark_stage(name, seconds=seconds, note=note)
    job.log(f"{name}: done in {seconds:.1f}s. {note}")
    out(f"  {name}: done in {seconds:.1f}s. {note}".rstrip())
    return "done"


def run_pipeline(cfg, job, start=None, out=print):
    """Run the stages in order. With `start`, that stage and everything after it run again.
    Stops at the first stage that is not built yet. Returns [(stage name, outcome), ...]."""
    order = names()
    first = 0
    if start is not None:
        get_stage(start)                       # a clear error for a typo
        first = order.index(start)
    results = []
    for index in range(first, len(order)):
        name = order[index]
        outcome = run_stage(cfg, job, name, force=(start is not None and index == first), out=out)
        results.append((name, outcome))
        if outcome == "not_built":
            out(f"Stopped: the '{name}' stage is not built yet.")
            break
    return results
