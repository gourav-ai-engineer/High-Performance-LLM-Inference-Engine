from __future__ import annotations

import asyncio
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


router = APIRouter()


@router.post("/v1/completions")
async def completions(payload: CompletionRequest, request: Request):
    engine = request.app.state.engine
    started = time.perf_counter()
    text = await engine.generate(payload.prompt, payload.max_tokens, payload.temperature, payload.top_p)
    return {"id": f"cmpl-{int(started * 1e6)}", "object": "text_completion", "choices": [CompletionChoice(text=text).model_dump()], "usage": {"completion_tokens": len(engine.tokenizer.encode(text, add_special_tokens=False))}}


@router.post("/v1/completions/stream")
async def completions_stream(payload: CompletionRequest, request: Request):
    engine = request.app.state.engine
    async def events():
        async for token in engine.generate_stream(payload.prompt, payload.max_tokens, payload.temperature, payload.top_p):
            yield f"data: {json.dumps({'choices': [{'text': token, 'index': 0, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/v1/models")
async def models(request: Request):
    return {"object": "list", "data": [{"id": request.app.state.config.model_name, "object": "model", "owned_by": "local"}]}


@router.get("/health")
async def health(request: Request):
    return {"status": "ok", "model": request.app.state.config.model_name}
