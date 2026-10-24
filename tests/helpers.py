"""Small builders shared by the stage tests: a settings dict that points at a temporary folder, with
prompt files in it, and a job that already holds the files an earlier stage would have made."""
from pathlib import Path

from core import job as jobmod

SIMPLIFY_PROMPT = "TEST SIMPLIFY PROMPT: rewrite in simple English."
DNT_PROMPT = "TEST DNT PROMPT: wrap names in <dnt> tags. Text to scan:"


def stage_cfg(tmp, **extra):
    """Settings for a stage test. tmp is a temporary folder path; jobs and prompts live inside it."""
    tmp = Path(tmp)
    prompts = tmp / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "simplify.txt").write_text(SIMPLIFY_PROMPT + "\n", encoding="utf-8")
    (prompts / "dnt.txt").write_text(DNT_PROMPT + "\n", encoding="utf-8")
    (prompts / "dnt_retry.txt").write_text("RETRY: {{PROBLEM}}\n", encoding="utf-8")
    cfg = {
        "paths": {"jobs": str(tmp / "jobs"), "prompts": str(prompts), "data": str(tmp / "data")},
        "models": {
            "local1": {"url": "http://fake:1234/v1", "model": "fake-model", "server": "lm_studio",
                       "temperature": 0.3, "max_tokens": 4096, "ttl": 600, "unload_on_switch": False},
            "local2": {"url": "http://fake:1234/v1", "model": "fake-model-2", "server": "lm_studio",
                       "unload_on_switch": False},
            "cloud1": {"url": "http://fake-cloud/v1", "model": "judge-model", "api_key_env": "TEST_CLOUD_KEY",
                       "temperature": 0.0, "max_tokens": 2048},
        },
        "stages": {
            "ingest": {"max_chars": 12000},
            "simplify": {"model": "local1", "retries": 0},
            "tag": {"model": "local1", "retries": 0, "on_fail": "continue"},
            "translate": {"model": "local1", "passes": 3, "temperature": 0.3, "pass_temp_step": 0.3},
            "judge_translation": {"model": "cloud1", "temperature": 0.0, "retries": 1},
            "revert": {"model": "local1", "passes": 3, "temperature": 0.3, "pass_temp_step": 0.3},
            "judge_comprehension": {"model": "cloud1", "temperature": 0.0, "retries": 1},
        },
        "translation": {"target_new_chars": 50, "revert_tolerance": 10},
    }
    cfg.update(extra)
    return cfg


def job_with(cfg, **files):
    """A new job whose folder already holds the named files, e.g. job_with(cfg, **{"01_clean.txt": text})."""
    job = jobmod.new_job(cfg, "source text for the test")
    for name, text in files.items():
        job.write(name.replace("__", "."), text)
    return job
