"""Test-run defaults for the environment server/main.py reads at import.

Two things this has to do, both because `server/main.py` calls `load_dotenv()`
at import (main.py:44) and will happily pick up a real deployment's `.env`.

`JWT_SECRET`: main.py:93 refuses to import without it, which is right for a
deployment and useless for a test run. A throwaway one means `pytest tests/`
works with no ceremony, in CI and on a laptop alike.

`POSTGRES_HOST`: pinned to loopback. Without this a developer with a populated
`.env` runs the suite against whatever host it names — I watched it dial a
production database. `load_dotenv()` does not override variables that are
already set, so claiming the name here is what keeps the suite local. Point it
somewhere real by exporting it yourself, deliberately, before the run.

`setdefault`, never assignment: an explicit export still wins. Nothing here is a
credential; the values never leave the test process.

`AUTH_DISABLED`: `server/auth.py` reads it once at import and every test module
shares that one module object, so the first import decides the mode for the
whole run — and `test_api_keys_router.py` imports `auth` at module level, which
sorts first. Deciding it here, before any test module loads, is what makes the
server tests pass together rather than only one file at a time. The one suite
that wants real auth, `test_server_auth.py`, is opt-in gated and will need its
`_load_app` to reload `auth` as well as `main` before it can run.
"""

import os

os.environ.setdefault("JWT_SECRET", "test-only-not-a-real-secret")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("AUTH_DISABLED", "true")
