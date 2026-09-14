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
    def _fused_gemm(A, W, S, C, M, N, K, stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            wq = tl.load(W + rk[:, None] * stride_wk + rn[None, :] * stride_wn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0)
            scale = tl.load(S + rk[:, None], mask=rk[:, None] < K, other=1.0)
            acc += tl.dot(a.to(tl.float16), (wq.to(tl.float16) * scale).to(tl.float16))
        tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fused_gemm(a: torch.Tensor, packed_weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not all(t.is_cuda for t in (a, packed_weight, scales)):
        raise ValueError("fused_gemm requires CUDA tensors")
    if packed_weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("packed_weight must be int8 or uint8")
    m, k = a.shape
    kw, n = packed_weight.shape
    if kw != k:
        raise ValueError("weight K dimension must match A")
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    _fused_gemm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](a, packed_weight, scales, c, m, n, k, a.stride(0), a.stride(1), packed_weight.stride(0), packed_weight.stride(1), c.stride(0), c.stride(1), BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    return c
