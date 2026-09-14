from __future__ import annotations

from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from fastapi.responses import Response

from engine.config import EngineConfig
from engine.model_loader import ModelLoader
from engine.block_manager import BlockManager
from engine.scheduler import ContinuousBatchingScheduler, Sequence
from telemetry.prometheus_exporter import metrics_payload, kv_cache_blocks_free
from .routes import router


class InferenceService:
    def __init__(self, bundle, config):
        self.bundle, self.config = bundle, config
        self.tokenizer = bundle.tokenizer
        self.block_manager = BlockManager(1024, config.block_size)
        self.scheduler = ContinuousBatchingScheduler(config.max_batch_size, self.block_manager, config.preemption_mode)

    async def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        ids = inputs.input_ids.to(next(self.bundle.model.parameters()).device)
        output = self.bundle.model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    async def generate_stream(self, prompt, max_tokens, temperature, top_p):
        text = await self.generate(prompt, max_tokens, temperature, top_p)
        for token in text.split():
            yield token + " "


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = EngineConfig()
    bundle = ModelLoader(config).load()
    app.state.config = config
    app.state.engine = InferenceService(bundle, config)
    kv_cache_blocks_free.set(app.state.engine.block_manager.get_num_free_blocks())
    yield


app = FastAPI(title="High-Performance LLM Inference Engine", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/metrics")
def metrics():
    return Response(metrics_payload(), media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
