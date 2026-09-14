# High-Performance LLM Inference Engine

A research-grade serving engine for studying the systems bottlenecks behind large-language-model inference: paged KV memory, continuous batching, preemption, quantization, speculative decoding, GPU kernels, and production telemetry.

> **Status:** v2 engineering pass. The project distinguishes CPU-safe scheduling/memory tests from CUDA/Triton integration paths; performance numbers are intentionally reported only after reproducible GPU benchmarks.

## Architecture
```text
Client -> OpenAI-compatible FastAPI -> Admission / Scheduler
                                      |
                    +-----------------+----------------+
                    |                                  |
               Waiting Queue                     Running Batch
                    |                                  |
              Block Manager -> Paged KV Cache -> Model Executor
                    |                                  |
              Prefix/COW memory                  Triton / PyTorch
                    |                                  |
              Prometheus -----------------------> Grafana
```

## Engineering Features
- Paged KV-cache allocation with fixed-size physical blocks
- Reference-counted copy-on-write sharing for reusable prefixes
- Iteration-level continuous batching with priority-aware admission
- Recompute/swap preemption primitives and cancellation handling
- Hugging Face model loading with configurable dtype and quantization
- INT8/INT4/FP8 quantization primitives
- Speculative draft/verify decoding primitives
- Triton paged-attention and fused dequantization/GEMM kernels
- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- SSE streaming with request IDs and cache-control headers
- Prometheus metrics + Grafana dashboard
- Reproducible chatbot, summarization and code-generation workloads
- CI quality gate with Ruff, Black and Pytest

## Quickstart

### Docker + NVIDIA GPU
```bash
docker compose up --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain PagedAttention briefly."}],"max_tokens":64}'
```

### Local development
```bash
python -m pip install -e '.[dev]'
cp .env.example .env
pytest -q
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Benchmarking

The benchmark suite measures TTFT, TPOT, end-to-end latency, p50/p95/p99 percentiles, throughput and GPU memory/utilization. Workloads are defined in `benchmarks/workloads.py` and results are recorded in `benchmarks/benchmark_report.md`.

```bash
./scripts/run_benchmarks.sh
python -m benchmarks.benchmark_serving --url http://localhost:8000 --requests 100 --concurrency 16
```

For credible comparisons, run the same model, GPU, dtype, workload, concurrency and warm-up policy against this engine, Transformers and vLLM. Do not infer speedups from theoretical targets.

## API

- `POST /v1/completions` — text completion
- `POST /v1/completions/stream` — SSE text completion
- `POST /v1/chat/completions` — OpenAI-compatible chat completion
- `GET /v1/models` — model discovery
- `GET /health` — health probe
- `GET /metrics` — Prometheus metrics

## Performance goals

The research plan targets lower KV fragmentation, higher sustained decode throughput and lower tail latency. The project treats these as **experiment hypotheses**, not pre-measured claims; actual gains must be established on controlled hardware. The original design calls for tracking TTFT, TPOT, throughput, GPU utilization, VRAM and p95/p99 latency.

## Roadmap

1. Baseline and reproducible benchmark suite
2. Continuous batching + paged KV memory
3. Quantization + kernel profiling
4. Prefix caching + chunked prefill
5. Speculative decoding acceptance-rate experiments
6. Disaggregated prefill/decode research
7. Dynamic LoRA serving

## Tech Stack

Python, PyTorch, Transformers, FastAPI, asyncio, Pydantic, Triton, bitsandbytes, Prometheus, Grafana, Docker/CUDA, aiohttp and Rich.

## Design references

The architecture studies ideas from vLLM, SGLang and TensorRT-LLM while keeping the implementation inspectable and benchmarkable end-to-end.
