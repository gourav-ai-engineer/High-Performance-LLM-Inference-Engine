from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class RequestMetrics:
    ttft: float
    tpot: float
    latency: float
    output_tokens: int

@dataclass
class BenchmarkResult:
    requests: list[RequestMetrics] = field(default_factory=list)

    def percentile(self, field: str, p: float) -> float:
        values = [getattr(x, field) for x in self.requests]
        return float(np.percentile(values, p)) if values else 0.0

    @property
    def throughput(self) -> float:
        total = sum(x.output_tokens for x in self.requests)
        duration = max((x.latency for x in self.requests), default=0.0)
        return total / duration if duration else 0.0
