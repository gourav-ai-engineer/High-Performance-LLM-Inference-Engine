from __future__ import annotations

import torch


class KVCache:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, dtype=torch.float16, device=None):
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int8):
            raise ValueError("unsupported KV cache dtype")
        self.num_blocks, self.block_size = num_blocks, block_size
        self.num_heads, self.head_dim = num_heads, head_dim
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.empty(shape, dtype=dtype, device=self.device)
        self.value_cache = torch.empty_like(self.key_cache)
        self._scale = 1.0

    def store(self, block_id: int, slot: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self._validate(block_id, slot)
        expected = (self.num_heads, self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"key/value must have shape {expected}")
        if self.dtype == torch.int8:
            scale = max(float(key.detach().abs().max()), float(value.detach().abs().max()), 1e-8) / 127.0
            self._scale = max(self._scale, scale)
            self.key_cache[block_id, slot].copy_((key / self._scale).round().clamp(-128, 127).to(torch.int8))
            self.value_cache[block_id, slot].copy_((value / self._scale).round().clamp(-128, 127).to(torch.int8))
        else:
            self.key_cache[block_id, slot].copy_(key.to(self.dtype))
            self.value_cache[block_id, slot].copy_(value.to(self.dtype))

    def get(self, block_id: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(block_id, slot)
        key = self.key_cache[block_id, slot]
        value = self.value_cache[block_id, slot]
        if self.dtype == torch.int8:
            return key.float() * self._scale, value.float() * self._scale
        return key, value

    def _validate(self, block_id: int, slot: int) -> None:
        if not 0 <= block_id < self.num_blocks or not 0 <= slot < self.block_size:
            raise IndexError("KV cache block or slot out of range")
