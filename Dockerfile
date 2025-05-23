FROM python:3.10-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential python3-dev git vim poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/hubl1/baryon-analysis"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install/deps /usr/local
COPY codes/ .

COPY requirements.txt .
COPY README.md .
LABEL license="MIT"

