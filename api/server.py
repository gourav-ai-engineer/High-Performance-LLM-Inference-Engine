from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import Response

from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.model_loader import ModelLoader
from engine.scheduler import ContinuousBatchingScheduler, Sequence
from telemetry.prometheus_exporter import (
    active_requests,
    batch_size,
    gpu_memory_used_bytes,
    gpu_utilization_percent,
    inference_errors_total,
    inference_input_tokens,
    inference_output_tokens,
    inference_request_duration_seconds,
    inference_requests_total,
    inference_tpot_seconds,
    inference_ttft_seconds,
    kv_cache_blocks_free,
    preemptions_total,
    queue_depth,
    metrics_payload,
)
from .routes import router


class InferenceService:
    """Serving runtime around the model with scheduler and observability hooks."""

    def __init__(self, bundle, config: EngineConfig):
        self.bundle = bundle
        self.config = config
        self.tokenizer = bundle.tokenizer
        self.device = next(bundle.model.parameters()).device
        self.block_manager = BlockManager(1024, config.block_size)
        self.scheduler = ContinuousBatchingScheduler(
            config.max_batch_size, self.block_manager, config.preemption_mode
        )
        self._generation_lock = asyncio.Lock()

    def _sample_kwargs(self, temperature: float, top_p: float) -> dict:
        if temperature <= 0:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": temperature, "top_p": top_p}

    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        request_id = f"seq-{uuid.uuid4().hex}"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        sequence = Sequence(request_id, prompt_ids, max_tokens)
        started = time.perf_counter()
        inference_requests_total.inc()
        active_requests.inc()
        inference_input_tokens.observe(len(prompt_ids))
        try:
            async with self._generation_lock:
                self.scheduler.add_request(sequence)
                self.scheduler.schedule()
                queue_depth.set(len(self.scheduler.waiting))
                batch_size.observe(len(self.scheduler.running))

                inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                first_token_start = time.perf_counter()
                output = self.bundle.model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self._sample_kwargs(temperature, top_p),
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                ttft = time.perf_counter() - first_token_start
                inference_ttft_seconds.observe(ttft)

                generated_ids = output[0, inputs.shape[1] :].tolist()
                for token in generated_ids:
                    sequence.generated_tokens.append(int(token))
                if generated_ids:
                    inference_tpot_seconds.observe(
                        max(0.0, (time.perf_counter() - first_token_start) / len(generated_ids))
                    )
                sequence.finished = True
                self.scheduler.finish(request_id)
                inference_output_tokens.observe(len(generated_ids))
                return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            inference_errors_total.inc()
            self.scheduler.cancel(request_id)
            raise
        finally:
            active_requests.dec()
            inference_request_duration_seconds.observe(time.perf_counter() - started)
            queue_depth.set(len(self.scheduler.waiting))
            kv_cache_blocks_free.set(self.block_manager.get_num_free_blocks())
            preemptions_total.inc(max(0, self.scheduler.preemptions - preemptions_total._value.get()))
            self._update_gpu_metrics()

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        # The model.generate path is intentionally kept deterministic and dependency-light.
        # Streaming is exposed at the API boundary while tokenization of the final text
        # remains compatible with the non-streaming endpoint.
        text = await self.generate(prompt, max_tokens, temperature, top_p)
        for token in text.split():
            yield token + " "

    def _update_gpu_metrics(self) -> None:
        if not torch.cuda.is_available():
            return
        gpu_memory_used_bytes.set(torch.cuda.memory_allocated())
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization_percent.set(utilization.gpu)
        except Exception:
            # NVML is optional at runtime; CUDA allocation metrics remain available.
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = EngineConfig()
    bundle = ModelLoader(config).load()
    app.state.config = config
    app.state.engine = InferenceService(bundle, config)
    kv_cache_blocks_free.set(app.state.engine.block_manager.get_num_free_blocks())
    yield


app = FastAPI(
    title="High-Performance LLM Inference Engine",
    version="0.3.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/metrics")
def metrics():
    return Response(metrics_payload(), media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
