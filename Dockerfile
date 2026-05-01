# syntax=docker/dockerfile:1
#
#   docker build -t cdss-nlp .                  # runtime image
#   docker build --target test -t cdss-nlp:test .   # runs the test suite during build
#
# UMLS data and the QuickUMLS index are NOT baked in (licence forbids
# redistribution). Mount them at runtime; see compose.yaml.

ARG PYTHON_VERSION=3.11

# ---- builder: compile deps into a venv -------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv $VIRTUAL_ENV

WORKDIR /app
# Install dependencies against a stub package first, so editing code doesn't
# invalidate this (slow) layer. Editable install => configs/ and data/kb/
# resolve relative to /app, exactly like a local checkout.
COPY pyproject.toml README.md ./
RUN mkdir -p src/medinfer && touch src/medinfer/__init__.py \
 && pip install --upgrade pip \
 && pip install -e .

# QuickUMLS downloads NLTK stopwords on first use; fetch them now so the
# container works offline. NLTK searches <sys.prefix>/nltk_data.
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/opt/venv/nltk_data', quiet=True)"

# ---- test: `docker build --target test .` --------------------------------
FROM builder AS test
RUN pip install pytest
COPY . .
RUN pytest -q

# ---- runtime ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# uid 1000 matches the usual host user, so bind-mounted results stay yours.
RUN useradd --create-home --uid 1000 medinfer

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY configs/ configs/
COPY src/ src/
COPY data/kb/ data/kb/
COPY data/processed/ data/processed/
COPY evaluation/ evaluation/
RUN mkdir -p data/quickumls_index evaluation/results \
 && chown medinfer data/quickumls_index evaluation/results

USER medinfer
ENTRYPOINT ["medinfer"]
CMD ["--help"]
