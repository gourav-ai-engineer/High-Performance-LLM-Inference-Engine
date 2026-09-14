from __future__ import annotations

import json
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .openai_models import ChatCompletionRequest


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


def _request_id(prefix: str = "cmpl") -> str:
    return f"{prefix}-{time.time_ns()}"


@router.post("/v1/completions")
async def completions(payload: CompletionRequest, request: Request):
    if payload.stream:
        raise HTTPException(status_code=400, detail="Use /v1/completions/stream for streaming")
    engine = request.app.state.engine
    text = await engine.generate(payload.prompt, payload.max_tokens, payload.temperature, payload.top_p)
    output_tokens = len(engine.tokenizer.encode(text, add_special_tokens=False))
    return {"id": _request_id(), "object": "text_completion", "choices": [CompletionChoice(text=text).model_dump()], "usage": {"prompt_tokens": len(engine.tokenizer.encode(payload.prompt, add_special_tokens=False)), "completion_tokens": output_tokens, "total_tokens": output_tokens}}


@router.post("/v1/completions/stream")
async def completions_stream(payload: CompletionRequest, request: Request):
    engine = request.app.state.engine

    async def events():
        async for token in engine.generate_stream(payload.prompt, payload.max_tokens, payload.temperature, payload.top_p):
            yield f"data: {json.dumps({'id': _request_id(), 'object': 'text_completion.chunk', 'choices': [{'text': token, 'index': 0, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    engine = request.app.state.engine
    text = await engine.generate(payload.prompt(), payload.max_tokens, payload.temperature, payload.top_p)
    return {"id": _request_id("chatcmpl"), "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {"completion_tokens": len(engine.tokenizer.encode(text, add_special_tokens=False))}}


@router.get("/v1/models")
async def models(request: Request):
    return {"object": "list", "data": [{"id": request.app.state.config.model_name, "object": "model", "owned_by": "local"}]}


@router.get("/health")
async def health(request: Request):
    return {"status": "ok", "model": request.app.state.config.model_name}
