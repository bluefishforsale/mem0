"""Pins the nightly wrapper's hold on the harness, which lives in a submodule this repo does not test."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("psycopg")


def harness_client(**kwargs):
    return sys.modules["benchmarks.common.llm_client"].LLMClient(**kwargs)


@pytest.fixture
def nightly(monkeypatch):
    class LLMClient:
        # The harness's signature at the pinned commit: timeout is a keyword default.
        def __init__(self, model="gpt-4o", provider="openai", rpm=200, timeout=120.0, **kwargs):
            self.timeout = timeout

    llm_client = types.ModuleType("benchmarks.common.llm_client")
    llm_client.LLMClient = LLMClient
    common = types.ModuleType("benchmarks.common")
    common.llm_client = llm_client
    monkeypatch.setitem(sys.modules, "benchmarks", types.ModuleType("benchmarks"))
    monkeypatch.setitem(sys.modules, "benchmarks.common", common)
    monkeypatch.setitem(sys.modules, "benchmarks.common.llm_client", llm_client)

    spec = importlib.util.spec_from_file_location("nightly", Path(__file__).with_name("nightly.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_default_timeout_outlasts_a_slow_local_judge(nightly):
    # 2026-11-12: ocean moved to a dense 27B that decodes at about 6 tokens per second and thinks first.
    # The harness's 120s default expired on most answers, each retried five times, and four nightly runs
    # in a row died at the six hour job limit having scored a third of the questions.
    nightly.give_the_judge_time()

    assert harness_client(model="judge", provider="openai", rpm=200).timeout == nightly.GENERATION_TIMEOUT_SECONDS
    assert nightly.GENERATION_TIMEOUT_SECONDS >= 900


def test_an_explicit_timeout_from_the_harness_still_wins(nightly):
    nightly.give_the_judge_time()

    assert harness_client(model="judge", timeout=30.0).timeout == 30.0
