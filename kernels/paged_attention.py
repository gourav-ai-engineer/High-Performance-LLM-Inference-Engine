from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _paged_attention(Q, K, V, O, table, n_tokens, stride_qb, stride_qh, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh, stride_ob, stride_oh, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs, mask=offs < HEAD_DIM, other=0.0)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        m = -float("inf")
        d = 0.0
        for pos in range(0, n_tokens):
            logical = pos // BLOCK_SIZE
            slot = pos % BLOCK_SIZE
            physical = tl.load(table + b * ((n_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) + logical)
            k = tl.load(K + physical * stride_kb + slot * stride_ks + h * stride_kh + offs, mask=offs < HEAD_DIM, other=0.0).to(tl.float32)
            v = tl.load(V + physical * stride_vb + slot * stride_vs + h * stride_vh + offs, mask=offs < HEAD_DIM, other=0.0).to(tl.float32)
            score = tl.sum(q.to(tl.float32) * k) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
            new_m = tl.maximum(m, score)
            alpha = tl.exp(m - new_m)
            beta = tl.exp(score - new_m)
            acc = acc * alpha + beta * v
            d = d * alpha + beta
            m = new_m
        out = acc / tl.maximum(d, 1e-6)
        tl.store(O + b * stride_ob + h * stride_oh + offs, out, mask=offs < HEAD_DIM)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, context_length: int, block_size: int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda:
        raise ValueError("paged_attention requires CUDA tensors")
    batch, heads, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (batch, heads)
    _paged_attention[grid](q, k_cache, v_cache, out, block_table, context_length, q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), out.stride(0), out.stride(1), BLOCK_SIZE=block_size, HEAD_DIM=head_dim)
    return out
