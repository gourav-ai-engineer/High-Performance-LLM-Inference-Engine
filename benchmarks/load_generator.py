from __future__ import annotations
import asyncio, random

async def poisson_arrivals(rate: float, count: int):
    if rate <= 0: raise ValueError("rate must be positive")
    for _ in range(count):
        await asyncio.sleep(random.expovariate(rate))
        yield

def synthetic_prompt(length: int) -> str:
    if length < 1: raise ValueError("length must be positive")
    return "token " * length
