#!/bin/sh
# Full local pipeline: unit tests, conformance check (local python), Docker build, and the
# conformance check inside the exact match sandbox (what the scorer runs before matches).
set -e
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-kestrel-7x}"

echo "== unit tests"
python3 -m unittest discover -s tests -q

echo "== conformance check (local python)"
python3 kit/check/check.py -- python3 -u bot.py

echo "== docker build"
docker build -t "$IMAGE" .

echo "== conformance check (match sandbox)"
python3 kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m \
  --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/kit/arena:/arena:ro" "$IMAGE"
