# Benchmark Report

This file is intentionally a measurement template: performance claims are populated only from recorded runs on a specified GPU/model configuration.

## Required experiment matrix

| Engine | Model | GPU | Prompt | Output | Concurrency | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Throughput | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| This engine | record run | record GPU | 256 | 128 | 1/8/32 | measured | measured | measured | measured | measured | measured |
| Transformers | same | same | 256 | 128 | 1/8/32 | measured | measured | measured | measured | measured | measured |
| vLLM | same | same | 256 | 128 | 1/8/32 | measured | measured | measured | measured | measured | measured |

## Workloads

- Interactive chatbot: short prompt and output; optimize TTFT.
- Summarization: 8k+ prompt; stress prefill and KV capacity.
- Code generation: medium prompt and long output; stress sustained decode.

## Reproducibility

Record model revision, GPU model, CUDA/PyTorch versions, dtype, quantization mode, block size, maximum context length, concurrency, random seed, and warm-up count with every benchmark result.

## Interpretation

Do not report a speedup unless both systems use the same model, precision, workload, hardware, and warm-up methodology. Report raw measurements alongside percentage deltas.
