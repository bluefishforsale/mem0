# The LOCOMO harness from the pinned evaluation/ submodule, the dataset, and the
# wrapper that turns a run into metrics.
FROM python:3.12-slim

WORKDIR /bench
COPY evaluation/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]>=3.2.8"

# Pinned to the commit that last touched the file, so the questions cannot
# change underneath the trend line.
ADD https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json /data/locomo10.json

COPY evaluation/ ./
COPY tests/quality/nightly/nightly.py ./nightly.py

CMD ["python", "nightly.py"]
