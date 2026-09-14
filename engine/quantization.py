from __future__ import annotations

import torch


class QuantizationEngine:
    SUPPORTED = {"int8", "int4", "fp8"}

    def quantize_weights(self, model, method: str):
        method = method.lower()
        if method not in self.SUPPORTED:
            raise ValueError(f"unsupported quantization method: {method}")
        if method in {"int8", "int4"}:
            try:
                from bitsandbytes.functional import quantize_4bit, quantize_8bit
            except ImportError as exc:
                raise RuntimeError("bitsandbytes is required for INT4/INT8 quantization") from exc
            for parameter in model.parameters():
                if parameter.ndim == 0 or not parameter.is_floating_point():
                    continue
                if method == "int4":
                    q, state = quantize_4bit(parameter.data, quant_type="nf4", compress_statistics=True)
                else:
                    q, state = quantize_8bit(parameter.data)
                parameter._quantized_data = q
                parameter._quantization_state = state
            return model
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("FP8 weight quantization requires a compatible NVIDIA GPU")
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype in (torch.float16, torch.bfloat16, torch.float32):
                parameter.data = parameter.data.to(torch.float8_e4m3fn)
        return model
