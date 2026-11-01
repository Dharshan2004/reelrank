FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY reelrank ./reelrank
RUN uv sync --locked --no-editable --no-dev
ENV PATH="/app/.venv/bin:$PATH" \
    REELRANK_DATA_DIR=/app/data \
    REELRANK_ARTIFACTS_DIR=/app/artifacts
EXPOSE 43123
CMD ["python", "-m", "uvicorn", "reelrank.serve.app:app", "--host", "0.0.0.0", "--port", "43123"]
