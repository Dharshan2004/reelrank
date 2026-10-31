FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY reelrank ./reelrank
RUN pip install --no-cache-dir -e .

# Expect data/artifacts to be mounted or baked in a later stage
COPY artifacts ./artifacts
COPY data/processed ./data/processed

ENV PORT=43123
EXPOSE 43123
CMD ["python", "-m", "uvicorn", "reelrank.serve.app:app", "--host", "0.0.0.0", "--port", "43123"]
