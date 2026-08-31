FROM python:3.12-slim

WORKDIR /app

# Server dependencies first, so a change to mem0 does not rebuild them.
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mem0 from THIS repository, not from PyPI.
#
# NOTE: this is the whole reason the build context is the repository root
# rather than server/. Installing the published mem0ai here would mean the
# image ignores every change in mem0/ and the fork contributes nothing but
# server/ — which is exactly the state this replaced.
#
# Kept in packages/ so the source tree does not sit on sys.path at /app and
# shadow the installed distribution.
WORKDIR /app/packages
COPY pyproject.toml README.md LICENSE ./
COPY mem0 ./mem0
RUN pip install --no-cache-dir ".[nlp]"

# Bake the lemmatizer model in. mem0 downloads en_core_web_sm on first use when
# spaCy is present without it, and this service is deliberately LAN-only; a
# surprise outbound fetch on the first search is not acceptable. Without spaCy
# at all, lemmatize_for_bm25() returns text unchanged and BM25 stops matching
# across word forms.
RUN python -m spacy download en_core_web_sm

WORKDIR /app
COPY server .

# Fail the build, not the deploy. `docker build` exits 0 on an image whose
# native dependencies are missing: plain psycopg needs a system libpq that this
# base image lacks, so the container built green and then died on first import.
# This imports the dependency surface — anything needing a shared library or a
# package that did not install — and turns that into a red build.
#
# NOTE: deliberately not `import main`. main.py calls initialize_state() at
# import time, which constructs Memory, builds the embedder and connects to
# postgres, so it can only succeed against a live database with real
# credentials. That belongs in a compose smoke test, not in the build.
RUN python -c "\
import psycopg, psycopg_pool, fastapi, uvicorn, alembic, sqlalchemy; \
import mem0; \
from mem0.vector_stores.pgvector import PGVector; \
from mem0.utils.lemmatization import lemmatize_for_bm25; \
assert lemmatize_for_bm25('memories') != '', 'lemmatizer returned nothing'; \
print('dependency surface OK')"

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# No --reload. It is a development reloader that restarts the app on any stat
# blip, and it has no business in a deployed image.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
