from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import EngineConfig


class ModelBundle:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


class ModelLoader:
    def __init__(self, config: EngineConfig):
        self.config = config

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return self.config.device

    def load(self) -> ModelBundle:
        device = self._resolve_device()
        dtype_name = self.config.dtype
        if device == "cpu" and dtype_name in {"float16", "bfloat16"}:
            dtype_name = "float32"
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]

        if self.config.quantization != "none" and device != "cuda":
            raise RuntimeError("INT4/INT8 quantization requires a CUDA device in this runtime")

        quant = None
        if self.config.quantization == "int8":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization == "int4":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        elif self.config.quantization == "fp8":
            raise NotImplementedError("FP8 loading requires a backend-specific quantization path")

        kwargs = {"torch_dtype": dtype, "device_map": "auto" if device == "cuda" else None}
        if quant is not None:
            kwargs["quantization_config"] = quant
        if device == "cpu":
            kwargs.pop("device_map")

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        if device == "cpu":
            model.to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return ModelBundle(model, tokenizer)
