"""Pipeline stages. Each module here is one step and follows one contract:

    def run(cfg, job) -> str

It reads the files it needs from the job folder (job.read), writes its outputs there (job.write),
and returns a one-line note for job.json and the terminal. It never calls another stage, and it
reaches a model only through core.llm. pipeline.py decides the order and what to skip.
"""
