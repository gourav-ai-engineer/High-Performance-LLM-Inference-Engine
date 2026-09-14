from __future__ import annotations

import torch


class ModelExecutor:
    def __init__(self, model_bundle, block_manager=None, kv_cache=None):
        self.model = model_bundle.model
        self.tokenizer = model_bundle.tokenizer
        self.block_manager = block_manager
        self.kv_cache = kv_cache
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def execute_prefill(self, sequences):
        if not sequences:
            return []
        encoded = [s.prompt_tokens for s in sequences]
        max_len = max(map(len, encoded))
        input_ids = torch.full((len(encoded), max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(encoded):
            input_ids[i, :len(ids)] = torch.tensor(ids, device=self.device)
            mask[i, :len(ids)] = 1
        out = self.model(input_ids=input_ids, attention_mask=mask, use_cache=True)
        return out

    @torch.inference_mode()
    def execute_decode(self, sequences):
        if not sequences:
            return {}
        ids = [s.generated_tokens[-1] if s.generated_tokens else s.prompt_tokens[-1] for s in sequences]
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(1)
        out = self.model(input_ids=input_ids, use_cache=True)
        return {s.sequence_id: int(torch.argmax(out.logits[i, -1]).item()) for i, s in enumerate(sequences)}
