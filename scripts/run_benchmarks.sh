#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 > /tmp/inference-engine.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -fsS http://127.0.0.1:8000/health >/dev/null && break; sleep 1; done
curl -fsS http://127.0.0.1:8000/health >/dev/null
python -m benchmarks.benchmark_serving --url http://127.0.0.1:8000/v1/completions --requests 20 --concurrency 4
