from __future__ import annotations

import torch


class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, tokenizer, k: int = 5):
        if k <= 0:
            raise ValueError("k must be positive")
        self.draft_model, self.target_model, self.tokenizer, self.k = draft_model, target_model, tokenizer, k

    @torch.inference_mode()
    def speculate(self, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids.clone()
        proposed = []
        for _ in range(self.k):
            logits = self.draft_model(ids).logits[:, -1, :]
            token = int(torch.argmax(logits, dim=-1).item())
            proposed.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
        return proposed

    @torch.inference_mode()
    def verify(self, input_ids: torch.Tensor, proposed: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not proposed:
            return torch.empty((0,), dtype=torch.long, device=input_ids.device), torch.empty((0,), device=input_ids.device)
        suffix = torch.tensor([proposed], dtype=torch.long, device=input_ids.device)
        logits = self.target_model(torch.cat([input_ids, suffix], dim=1)).logits[:, -len(proposed)-1:-1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return torch.tensor(proposed, device=input_ids.device), probs

    def accept_or_reject(self, proposed: torch.Tensor, target_probs: torch.Tensor, draft_probs: torch.Tensor | None = None) -> list[int]:
        accepted: list[int] = []
        for i, token in enumerate(proposed.tolist()):
            if i >= target_probs.shape[0]:
                break
            if draft_probs is None:
                accepted.append(token if int(torch.argmax(target_probs[i]).item()) == token else int(torch.argmax(target_probs[i]).item()))
                if accepted[-1] != token:
                    break
                continue
            p = float(target_probs[i, token])
            q = max(float(draft_probs[i, token]), 1e-8)
            if torch.rand((), device=target_probs.device).item() <= min(1.0, p / q):
                accepted.append(token)
            else:
                residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                residual = residual / residual.sum().clamp_min(1e-8)
                accepted.append(int(torch.multinomial(residual, 1).item()))
                break
        return accepted
