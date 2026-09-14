from __future__ import annotations

from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_name: str = "sshleifer/tiny-gpt2"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    quantization: Literal["none", "int8", "int4", "fp8"] = "none"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    enable_speculative: bool = False
    draft_model_name: str | None = None
    speculative_k: int = 5
    max_batch_size: int = 32
    preemption_mode: Literal["recompute", "swap"] = "recompute"
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("block_size")
    @classmethod
    def power_of_two(cls, value: int) -> int:
        if value < 1 or value & (value - 1):
            raise ValueError("block_size must be a positive power of two")
        return value

    @field_validator("gpu_memory_utilization")
    @classmethod
    def memory_fraction(cls, value: float) -> float:
        if not 0.1 <= value <= 0.99:
            raise ValueError("gpu_memory_utilization must be between 0.1 and 0.99")
        return value

    @field_validator("max_model_len", "max_batch_size", "speculative_k")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value
