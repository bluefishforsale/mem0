"""What the server's config state must survive.

`update_config` rebuilds the Memory instance from a merged config. That build
can reject the merge — an out-of-range knob always could, and since
MemoryConfig forbids unknown keys, so can a typo. The question these pin is
what the server is left holding when it does.

NOTE: imports the flat `server_state`, not `server.server_state`. `server/`
is on the path as its own root (main.py does `from server_state import ...`),
so the two names are separate module objects with separate globals, and
patching one leaves the handler using the other.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
server_state = pytest.importorskip("server_state", reason="server/ not on sys.path")

from fastapi import HTTPException  # noqa: E402

os.environ.setdefault("OPENAI_API_KEY", "fake-key")

VALID_CONFIG = {
    "version": "v1.1",
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
    "history_db_path": ":memory:",
}

# server/main.py builds a Memory at import time, so it only imports with a
# mocked backend. Skip rather than fail where it will not import at all; the
# SDK's CI installs the server's dependencies only via the server-tests extra.
#
# NOTE: AUTH_DISABLED must match what test_server_params.py sets, because
# server/auth.py reads it once at import and both files share that one module
# object. Whoever imports first decides the mode for both, so the two must
# agree or the pair passes alone and fails together. test_server_auth.py wants
# the opposite mode and gets away with it only because it skips without a
# database — giving it one means teaching its `_load_app` to reload `auth`,
# not just `main`.
with patch.dict(os.environ, {"AUTH_DISABLED": "true"}):
    with patch("mem0.Memory.from_config", return_value=MagicMock()):
        server_main = pytest.importorskip("main", reason="server/main.py not importable here")


@pytest.fixture
def initialized_state():
    """A server holding VALID_CONFIG, with no override row and no real backend."""
    with (
        patch.object(server_state, "_load_overrides", return_value={}),
        patch.object(server_state, "_save_overrides"),
        patch("mem0.Memory.from_config", return_value=MagicMock()) as from_config,
    ):
        server_state.initialize_state(VALID_CONFIG)
        yield from_config


def test_a_rejected_update_leaves_the_live_config_alone(initialized_state):
    """`_current_config` was assigned before the build that can raise, so a
    rejected update stuck as the base for the next merge while the running
    Memory kept serving the old one. The next valid update then inherited the
    bad key and failed too, and nothing in the config the server reported back
    explained why.
    """
    good = server_state.get_current_config()

    initialized_state.side_effect = ValueError("nope")
    with pytest.raises(ValueError):
        server_state.update_config({"add_context_topk": 3})

    assert server_state.get_current_config() == good


def test_a_rejected_update_keeps_the_running_memory(initialized_state):
    """The other half: the instance that was serving requests must survive a
    failed reconfigure, not be replaced by a half-built one.
    """
    live = server_state.get_memory_instance()

    initialized_state.side_effect = ValueError("nope")
    with pytest.raises(ValueError):
        server_state.update_config({"add_context_topk": 3})

    assert server_state.get_memory_instance() is live


def test_configure_answers_a_bad_key_with_a_client_error():
    """A config the server cannot use is the caller's mistake, so it gets a 4xx
    naming the key — the same answer `_validate_bundled_providers` already gives
    for an unbundled provider. Without this the ValidationError escapes the
    handler as an unhandled 500, which reads as "the server broke" rather than
    "your key is wrong".

    The rejection itself is unmocked: MemoryConfig does it, and it raises before
    any backend gets built, so the handler never needs a live store.
    """
    with (
        patch.object(server_state, "_load_overrides", return_value={}),
        patch.object(server_state, "_save_overrides"),
        patch("mem0.Memory.from_config", return_value=MagicMock()),
    ):
        server_state.initialize_state(VALID_CONFIG)

    with pytest.raises(HTTPException) as excinfo:
        server_main.set_config({"add_context_topk": 3}, _auth=None)

    assert excinfo.value.status_code == 400
    assert "add_context_topk" in str(excinfo.value.detail)
