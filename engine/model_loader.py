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

    def load(self) -> ModelBundle:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.config.dtype]
        quant = None
        if self.config.quantization == "int8":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization == "int4":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        if quant is not None:
            kwargs["quantization_config"] = quant
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return ModelBundle(model, tokenizer)
