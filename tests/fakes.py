"""A scripted stand-in for the model server, so stages can be tested with no LM Studio and no cloud key.

    from tests.fakes import FakeLLM

    with FakeLLM("first reply", "second reply") as fake:        # replies come back in order
        text = chat(cfg, "prompt", "story")
    fake.calls[0]["body"]["temperature"]                        # what the code actually sent

    with FakeLLM(handler=lambda body, n: f"call {n}"):          # or compute each reply
        ...

A scripted item may be a string (the reply text), an Exception (raised, like a network failure), or a
full response dict (to test odd replies, such as finish_reason "length").
"""
import io
import urllib.error
from unittest import mock


def reply_json(text, finish="stop", tokens=None):
    """The shape of an OpenAI-compatible chat completion."""
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"completion_tokens": tokens if tokens is not None else max(1, len(text) // 2)}}


def http_error(code=500, body="server error"):
    return urllib.error.HTTPError("http://fake", code, "error", {}, io.BytesIO(body.encode("utf-8")))


class FakeLLM:
    def __init__(self, *replies, handler=None, models=None):
        self.replies = list(replies)
        self.handler = handler
        self.models = models or ["fake-model"]
        self.calls = []                    # one dict per request: url, body, headers, timeout
        self._patches = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        if self.handler:
            item = self.handler(body, len(self.calls) - 1)
        elif self.replies:
            item = self.replies.pop(0)
        else:
            raise AssertionError("FakeLLM was called more times than it has scripted replies")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return item
        return reply_json(item)

    def _models(self, url, headers, timeout):
        return {"data": [{"id": m} for m in self.models]}

    def __enter__(self):
        for target, fn in (("core.llm._post", self), ("core.llm._get", self._models)):
            p = mock.patch(target, fn)
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._patches = []
        return False
