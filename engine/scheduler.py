from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Sequence:
    sequence_id: str
    prompt_tokens: list[int]
    max_new_tokens: int
    priority: int = 0
    generated_tokens: list[int] = field(default_factory=list)
    finished: bool = False
    preempted: bool = False
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def remaining(self) -> int:
        return max(0, self.max_new_tokens - len(self.generated_tokens))


class ContinuousBatchingScheduler:
    """Iteration-level scheduler with priority, admission control and preemption."""

    def __init__(self, max_batch_size: int, block_manager, preemption_mode: str = "recompute"):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if preemption_mode not in {"recompute", "swap"}:
            raise ValueError("preemption_mode must be recompute or swap")
        self.max_batch_size = max_batch_size
        self.block_manager = block_manager
        self.preemption_mode = preemption_mode
        self.waiting: deque[Sequence] = deque()
        self.running: dict[str, Sequence] = {}
        self.preemptions = 0

    def add_request(self, sequence: Sequence) -> None:
        if sequence.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if sequence.sequence_id in self.running or any(s.sequence_id == sequence.sequence_id for s in self.waiting):
            raise ValueError(f"duplicate sequence id: {sequence.sequence_id}")
        self.waiting.append(sequence)

    def cancel(self, sequence_id: str) -> bool:
        seq = self.running.pop(sequence_id, None)
        if seq is not None:
            seq.cancelled = True
            seq.finished = True
            self.block_manager.free(sequence_id)
            return True
        for seq in list(self.waiting):
            if seq.sequence_id == sequence_id:
                self.waiting.remove(seq)
                seq.cancelled = True
                seq.finished = True
                return True
        return False

    def finish(self, sequence_id: str) -> None:
        sequence = self.running.pop(sequence_id, None)
        if sequence:
            sequence.finished = True
            self.block_manager.free(sequence_id)

    def schedule(self) -> list[Sequence]:
        self._remove_finished()
        while len(self.running) < self.max_batch_size and self.waiting:
            candidate = max(self.waiting, key=lambda s: (s.priority, -s.created_at))
            self.waiting.remove(candidate)
            try:
                self.block_manager.allocate(candidate.sequence_id, max(1, candidate.total_tokens))
                candidate.preempted = False
                self.running[candidate.sequence_id] = candidate
            except MemoryError:
                candidate.preempted = True
                if self.preemption_mode == "swap" and self.running:
                    victim = min(self.running.values(), key=lambda s: (s.priority, s.created_at))
                    if victim.priority < candidate.priority:
                        self._preempt(victim)
                        self.waiting.appendleft(candidate)
                        continue
                self.waiting.appendleft(candidate)
                break
        return list(self.running.values())

    def step(self, generated: dict[str, int]) -> list[Sequence]:
        for sequence_id, token in generated.items():
            seq = self.running.get(sequence_id)
            if seq is None or seq.cancelled:
                continue
            seq.generated_tokens.append(int(token))
            self.block_manager.allocate(sequence_id, seq.total_tokens)
            if seq.remaining == 0:
                self.finish(sequence_id)
        return self.schedule()

    def _preempt(self, sequence: Sequence) -> None:
        self.running.pop(sequence.sequence_id, None)
        self.block_manager.free(sequence.sequence_id)
        sequence.preempted = True
        self.preemptions += 1
        self.waiting.append(sequence)

    def _remove_finished(self) -> None:
        for seq in list(self.running.values()):
            if seq.finished or seq.cancelled:
                self.finish(seq.sequence_id)
