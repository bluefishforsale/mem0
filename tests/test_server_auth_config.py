"""Auth configuration must be read when a request is judged, not when the
module happens to be imported.

`server/auth.py` read AUTH_DISABLED, ADMIN_API_KEY and JWT_SECRET into module
constants at import. Every test module shares that one module object, so the
first import decided the auth mode for the whole run: `test_api_keys_router.py`
imports `auth` at module level and sorts first, which is why the server tests
passed one file at a time and failed together. `_load_app` reloading `main` was
never enough, because the constants live in `auth`.

These do not need a database: the branches under test return before any query.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
auth = pytest.importorskip("auth", reason="server/ not on sys.path")

from fastapi import HTTPException  # noqa: E402


def _judge():
    """Run verify_auth with no bearer token and no API key."""
    request = SimpleNamespace(state=SimpleNamespace())
    return asyncio.run(auth.verify_auth(request=request, credentials=None, x_api_key=None))


def test_auth_disabled_is_read_when_the_request_is_judged():
    """Flipping the variable after import has to change the answer. It did not,
    which is the whole reason conftest had to pick a mode for the entire suite.
    """
    with patch.dict(os.environ, {"AUTH_DISABLED": "true"}):
        assert _judge() is None, "AUTH_DISABLED=true should let an anonymous request through"

    with patch.dict(os.environ, {"AUTH_DISABLED": "false"}):
        with pytest.raises(HTTPException) as excinfo:
            _judge()
        assert excinfo.value.status_code == 401
