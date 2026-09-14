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
    created_at: float = field(default_factory=time.monotonic)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def remaining(self) -> int:
        return max(0, self.max_new_tokens - len(self.generated_tokens))


class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int, block_manager, preemption_mode: str = "recompute"):
        self.max_batch_size = max_batch_size
        self.block_manager = block_manager
        self.preemption_mode = preemption_mode
        self.waiting: deque[Sequence] = deque()
        self.running: dict[str, Sequence] = {}

    def add_request(self, sequence: Sequence) -> None:
        if sequence.sequence_id in self.running or any(s.sequence_id == sequence.sequence_id for s in self.waiting):
            raise ValueError(f"duplicate sequence id: {sequence.sequence_id}")
        self.waiting.append(sequence)

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
                    victim = min(self.running.values(), key=lambda s: (s.priority, -s.created_at))
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
            if seq is None:
                continue
            seq.generated_tokens.append(int(token))
            if seq.remaining == 0:
                self.finish(sequence_id)
        return self.schedule()

    def _preempt(self, sequence: Sequence) -> None:
        self.running.pop(sequence.sequence_id, None)
        self.block_manager.free(sequence.sequence_id)
        sequence.preempted = True
        self.waiting.append(sequence)

    def _remove_finished(self) -> None:
        for seq in list(self.running.values()):
            if seq.finished:
                self.finish(seq.sequence_id)
