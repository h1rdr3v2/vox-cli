# vox on Linux: installs with plain pip (no uv) and runs the test suite.
# Build and run with scripts/test-linux.sh.
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install . pytest
COPY tests ./tests
CMD ["pytest", "-q", "-p", "no:warnings"]
