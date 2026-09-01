"""What must survive the removal of the notice machinery.

`reference_date` and `decay` are platform-only parameters that the OSS SDK
rejects. Their error text used to come from notices.py, which fetched a
PostHog feature-flag payload and fell back to a plain constant when telemetry
was off. These pin the rejection itself, so deleting that machinery cannot
quietly turn a rejected parameter into an ignored one.

`timestamp` is here as the counter-case: this fork supports it, so it must
stay ungated.
"""

import os
from unittest.mock import Mock, patch

import pytest

os.environ["OPENAI_API_KEY"] = "123"

from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.memory.main import Memory  # noqa: E402


@pytest.fixture
def memory_instance():
    with (
        patch("mem0.utils.factory.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.utils.factory.LlmFactory") as mock_llm,
        patch("mem0.memory.telemetry.capture_event"),
    ):
        mock_embedder.create.return_value = Mock()
        mock_vector_store.create.return_value = Mock()
        mock_vector_store.create.return_value.search.return_value = []
        mock_llm.create.return_value = Mock()
        return Memory(MemoryConfig(version="v1.1"))


def test_search_still_rejects_reference_date(memory_instance):
    with pytest.raises(ValueError, match="reference_date parameter is not supported"):
        memory_instance.search("q", filters={"user_id": "u"}, reference_date="2026-01-01")


def test_project_update_still_rejects_decay(memory_instance):
    with pytest.raises(ValueError, match="decay parameter is not supported"):
        memory_instance.project.update(decay=True)


@pytest.mark.parametrize("kwargs", [{}, {"decay": False}])
def test_project_update_without_decay_gives_the_generic_error(memory_instance, kwargs):
    """Only decay=True gets the decay-specific message. Everything else, including
    decay=False, is refused as an unsupported operation. Keeping the two apart
    matters: they tell the caller different things about what to change.
    """
    with pytest.raises(ValueError, match="Project updates are not supported"):
        memory_instance.project.update(**kwargs)


def test_add_timestamp_is_not_rejected(memory_instance):
    """This fork added backdated imports, so timestamp must not be gated. It
    still validates the value, which is why a bad one raises about ISO-8601
    rather than about support.
    """
    with pytest.raises(ValueError, match="ISO-8601"):
        memory_instance.add("hello", user_id="u", timestamp="not-a-date")
