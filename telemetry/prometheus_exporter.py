from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

REGISTRY = CollectorRegistry()
inference_requests_total = Counter("inference_requests_total", "Inference requests", registry=REGISTRY)
inference_ttft_seconds = Histogram("inference_ttft_seconds", "Time to first token", registry=REGISTRY)
inference_tpot_seconds = Histogram("inference_tpot_seconds", "Time per output token", registry=REGISTRY)
gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used", registry=REGISTRY)
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", registry=REGISTRY)
active_requests = Gauge("active_requests", "Active requests", registry=REGISTRY)
kv_cache_blocks_free = Gauge("kv_cache_blocks_free", "Free KV cache blocks", registry=REGISTRY)

def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)
