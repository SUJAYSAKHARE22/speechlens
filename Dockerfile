FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data/demo ./data/demo
RUN pip install --no-cache-dir ".[app]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["speechlens", "serve", "--data", "data/demo", "--host", "0.0.0.0", "--port", "8000"]
