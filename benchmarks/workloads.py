from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    output_tokens: int


WORKLOADS = (
    Workload("chatbot", 256, 128),
    Workload("summarization", 8192, 512),
    Workload("code-generation", 1024, 1024),
)


def synthetic_prompt(token_count: int) -> str:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    # Repetition gives deterministic, tokenizer-friendly benchmark input.
    return "performance benchmark token " * token_count
