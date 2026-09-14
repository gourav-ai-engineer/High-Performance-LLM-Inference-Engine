# High-Performance LLM Inference Engine

A systems-oriented LLM serving engine focused on KV-cache efficiency, continuous batching, quantization, speculative decoding, and GPU telemetry.

> **Status:** v1 research/engineering implementation. CUDA/Triton paths are enabled when compatible GPU dependencies are installed; CPU-safe unit tests cover memory-management primitives.

## Architecture
```text
Client -> FastAPI/SSE -> ContinuousBatchingScheduler
                         |             |
                    BlockManager   ModelExecutor
                         |             |
                    Paged KV Cache -> Triton kernels -> GPU
                         |
                 Prometheus -> Grafana
```

## Features
- Paged KV-cache allocation with fixed-size physical blocks
- Copy-on-write block sharing for prefix/beam-style workloads
- Iteration-level continuous batching and priority-aware preemption
- Hugging Face model loading with optional 4/8-bit quantization
- Speculative draft/verify decoding primitives
- Triton paged-attention and fused dequantization/GEMM kernels
- OpenAI-style completion API with SSE streaming
- Prometheus metrics and Grafana dashboard
- Async serving and internal benchmark harness

## Quickstart
Requirements: Python 3.11+, PyTorch, and an NVIDIA GPU for CUDA execution.

```bash
docker compose up --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/completions -H 'content-type: application/json' -d '{"prompt":"Explain KV caching in one paragraph.","max_tokens":32}'
```

Local development:
```bash
python -m pip install -e '.[dev]'
cp .env.example .env
uvicorn api.server:app --host 0.0.0.0 --port 8000
pytest -q
```

## Benchmarking
```bash
./scripts/run_benchmarks.sh
python -m benchmarks.benchmark_serving --url http://localhost:8000 --requests 100 --concurrency 16
```
Tracked metrics include TTFT, TPOT, end-to-end latency, throughput, active requests, KV-cache utilization, and GPU memory/utilization.

## API
`POST /v1/completions` accepts `prompt`, `max_tokens`, `temperature`, `top_p`, and `stream`.
`POST /v1/completions/stream` returns Server-Sent Events. `GET /v1/models`, `/health`, and `/metrics` expose discovery, health, and telemetry.

## Tech stack
Python, PyTorch, Transformers, FastAPI, asyncio, Pydantic, Triton, bitsandbytes, Prometheus, Grafana, Docker/CUDA, aiohttp and Rich.

## Design references
The implementation studies established ideas from vLLM, SGLang and TensorRT-LLM while keeping the engine intentionally small enough to benchmark and inspect end-to-end.
