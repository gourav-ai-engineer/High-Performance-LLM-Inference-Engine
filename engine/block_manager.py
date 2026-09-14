from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from threading import RLock


@dataclass
class BlockRef:
    block_id: int
    ref_count: int = 1


class BlockManager:
    """Owns the logical-to-physical mapping for paged KV cache blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = deque(range(num_blocks))
        self._tables: dict[str, list[BlockRef]] = {}
        self._lock = RLock()

    def allocate(self, sequence_id: str, num_tokens: int) -> list[int]:
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        needed = (num_tokens + self.block_size - 1) // self.block_size
        with self._lock:
            existing = self._tables.get(sequence_id, [])
            additional = max(0, needed - len(existing))
            if additional > len(self._free):
                raise MemoryError(f"KV cache exhausted: need {additional}, have {len(self._free)}")
            for _ in range(additional):
                existing.append(BlockRef(self._free.popleft()))
            self._tables[sequence_id] = existing
            return [r.block_id for r in existing]

    def free(self, sequence_id: str) -> None:
        with self._lock:
            refs = self._tables.pop(sequence_id, [])
            for ref in refs:
                ref.ref_count -= 1
                if ref.ref_count == 0:
                    self._free.append(ref.block_id)

    def copy_on_write(self, sequence_id: str, block_position: int) -> int:
        with self._lock:
            refs = self._tables.get(sequence_id)
            if refs is None or not 0 <= block_position < len(refs):
                raise IndexError("invalid sequence or block position")
            old = refs[block_position]
            if old.ref_count == 1:
                return old.block_id
            if not self._free:
                raise MemoryError("KV cache exhausted during copy-on-write")
            new = BlockRef(self._free.popleft())
            old.ref_count -= 1
            refs[block_position] = new
            return new.block_id

    def share(self, source_sequence_id: str, target_sequence_id: str) -> list[int]:
        with self._lock:
            if target_sequence_id in self._tables:
                raise ValueError("target sequence already exists")
            source = self._tables.get(source_sequence_id)
            if source is None:
                raise KeyError(source_sequence_id)
            self._tables[target_sequence_id] = [BlockRef(r.block_id, r.ref_count + 1) for r in source]
            for ref in source:
                ref.ref_count += 1
            return [r.block_id for r in source]

    def get_blocks(self, sequence_id: str) -> list[int]:
        with self._lock:
            return [r.block_id for r in self._tables.get(sequence_id, [])]

    def get_num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free)
