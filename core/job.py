"""Job folders: one folder per article under jobs/, holding every file the stages read and write.

    job = new_job(cfg, "article text ...")        # creates jobs/2026-10-17_0104_<slug>/ with 00_source.txt
    job = open_job(cfg, "2026-10-17_0104_<slug>") # an existing job
    job.write("01_clean.txt", text); job.read("01_clean.txt")
    job.mark_stage("ingest", seconds=0.4)         # recorded in job.json so a rerun can skip it
    job.stage_done("ingest")                      # True

job.json keeps what has finished, so a stage can be skipped or resumed. Writes are atomic: a crash
cannot leave half a file behind.
"""
import json
import os
import re
import time
from pathlib import Path

from core.config import ConfigError, resolve_path


def make_slug(text, words=6, limit=40):
    """A short folder-name label from the first words of the text."""
    slug = "_".join(re.findall(r"[A-Za-z0-9]+", text)[:words])[:limit]
    return slug or "story"


def _now_iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


class Job:
    def __init__(self, path):
        self.path = Path(path)

    @property
    def id(self):
        return self.path.name

    # ----- files -----

    def file(self, name):
        return self.path / name

    def exists(self, name):
        return self.file(name).exists()

    def read(self, name):
        try:
            return self.file(name).read_text(encoding="utf-8")
        except OSError:
            raise FileNotFoundError(f"{self.id}: {name} not found. Has the stage that writes it run?")

    def write(self, name, text):
        """Write text atomically (to a temporary file, then rename)."""
        target = self.file(name)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        return target

    def read_json(self, name):
        return json.loads(self.read(name))

    def write_json(self, name, data):
        return self.write(name, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    # ----- status -----

    def status(self):
        try:
            return self.read_json("job.json")
        except (FileNotFoundError, ValueError):
            return {"stages": {}}

    def stage_done(self, stage):
        return self.status().get("stages", {}).get(stage, {}).get("status") == "done"

    def done_stages(self):
        return [s for s, v in self.status().get("stages", {}).items() if v.get("status") == "done"]

    def mark_stage(self, stage, seconds=None, note=""):
        data = self.status()
        entry = {"status": "done", "finished": _now_iso()}
        if seconds is not None:
            entry["seconds"] = round(seconds, 1)
        if note:
            entry["note"] = note
        data.setdefault("stages", {})[stage] = entry
        data["last_finished"] = stage
        self.write_json("job.json", data)

    def mark_failed(self, stage, error):
        data = self.status()
        data.setdefault("stages", {})[stage] = {"status": "failed", "finished": _now_iso(), "error": str(error)}
        self.write_json("job.json", data)

    def set_meta(self, **values):
        """Free-form facts about the job (email subject, message ID, and so on)."""
        data = self.status()
        data.setdefault("meta", {}).update(values)
        self.write_json("job.json", data)

    def log(self, message):
        with open(self.file("log.txt"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


def new_job(cfg, source_text, slug=None, **meta):
    """Create a job folder holding the raw input as 00_source.txt. Never reuses a folder name."""
    jobs_dir = resolve_path(cfg, "jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y-%m-%d_%H%M')}_{slug or make_slug(source_text)}"
    path, n = jobs_dir / stem, 2
    while path.exists():
        path, n = jobs_dir / f"{stem}_{n}", n + 1
    path.mkdir()
    job = Job(path)
    job.write("00_source.txt", source_text)
    job.write_json("job.json", {"created": _now_iso(), "stages": {}, "meta": meta})
    job.log("job created")
    return job


def open_job(cfg, job_id):
    """An existing job, by folder name. Raises ConfigError (a plain message) if it is not there."""
    path = resolve_path(cfg, "jobs") / job_id
    if not path.is_dir():
        raise ConfigError(f"No job named {job_id}. Run list to see the jobs.")
    return Job(path)


def list_jobs(cfg):
    """All jobs, oldest first."""
    jobs_dir = resolve_path(cfg, "jobs")
    if not jobs_dir.is_dir():
        return []
    return [Job(p) for p in sorted(jobs_dir.iterdir()) if p.is_dir() and (p / "job.json").exists()]


def latest_job(cfg):
    jobs = list_jobs(cfg)
    return jobs[-1] if jobs else None
