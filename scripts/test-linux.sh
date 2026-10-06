#!/bin/sh
# Run vox's tests on Linux in Docker. On an Apple Silicon Mac this is
# Linux arm64; add --platform linux/amd64 for x86_64 (slower, emulated).
#
#   scripts/test-linux.sh [docker build options]
#
# Set VOX_INTEGRATION_MODELS to a models folder with Linux (portable) models
# to also run the speak-then-transcribe test.
set -e
cd "$(dirname "$0")/.."
docker build -f tests/linux/Dockerfile -t vox-linux-test "$@" .
if [ -n "$VOX_INTEGRATION_MODELS" ]; then
  exec docker run --rm -v "$VOX_INTEGRATION_MODELS:/models:ro" -e VOX_INTEGRATION_MODELS=/models vox-linux-test
fi
exec docker run --rm vox-linux-test
