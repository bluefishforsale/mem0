"""Test-run defaults for the environment server/main.py reads at import.

Both of these exist because `server/main.py` calls `load_dotenv()` at import
(main.py:44) and will happily pick up a real deployment's `.env`.

`JWT_SECRET`: main.py refuses to import without it, which is right for a
deployment and useless for a test run. A throwaway one means `pytest tests/`
works with no ceremony, in CI and on a laptop alike.

`POSTGRES_HOST`: pinned to loopback. Without this a developer with a populated
`.env` runs the suite against whatever host it names — I watched it dial a
production database. `load_dotenv()` does not override variables that are
already set, so claiming the name here is what keeps the suite local. Point it
somewhere real by exporting it yourself, deliberately, before the run.

`setdefault`, never assignment: an explicit export still wins. Nothing here is a
credential; the values never leave the test process.

NOTE: there is deliberately no AUTH_DISABLED default any more. There used to be,
because `server/auth.py` read the variable once at import and the first test
module to import it fixed the auth mode for the whole run. auth.py reads it per
request now, so each test file sets the mode it wants and they stop fighting.
Putting a default back here would hide the next regression of that kind.
"""

import os

os.environ.setdefault("JWT_SECRET", "test-only-not-a-real-secret")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
