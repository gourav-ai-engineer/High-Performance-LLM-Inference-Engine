from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

REGISTRY = CollectorRegistry()

inference_requests_total = Counter(
    "inference_requests_total", "Inference requests", registry=REGISTRY
)
inference_errors_total = Counter(
    "inference_errors_total", "Inference request failures", registry=REGISTRY
)
inference_ttft_seconds = Histogram(
    "inference_ttft_seconds", "Time to first token", registry=REGISTRY
)
inference_tpot_seconds = Histogram(
    "inference_tpot_seconds", "Time per output token", registry=REGISTRY
)
inference_request_duration_seconds = Histogram(
    "inference_request_duration_seconds", "End-to-end request duration", registry=REGISTRY
)
inference_input_tokens = Histogram(
    "inference_input_tokens", "Prompt token count", registry=REGISTRY
)
inference_output_tokens = Histogram(
    "inference_output_tokens", "Generated token count", registry=REGISTRY
)
batch_size = Histogram(
    "inference_batch_size", "Scheduler batch size", registry=REGISTRY
)
active_requests = Gauge("active_requests", "Active requests", registry=REGISTRY)
queue_depth = Gauge("inference_queue_depth", "Waiting scheduler requests", registry=REGISTRY)
preemptions_total = Counter("inference_preemptions_total", "KV-cache preemptions", registry=REGISTRY)
kv_cache_blocks_free = Gauge("kv_cache_blocks_free", "Free KV cache blocks", registry=REGISTRY)
gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used", registry=REGISTRY)
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", registry=REGISTRY)


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)
