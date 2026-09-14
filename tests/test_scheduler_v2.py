from engine.scheduler import ContinuousBatchingScheduler, Sequence


class FakeBlocks:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.allocations = {}

    def allocate(self, sequence_id, tokens):
        needed = (tokens + 3) // 4
        if needed > self.capacity - sum(self.allocations.values()):
            raise MemoryError
        self.allocations[sequence_id] = max(self.allocations.get(sequence_id, 0), needed)
        return list(range(self.allocations[sequence_id]))

    def free(self, sequence_id):
        self.allocations.pop(sequence_id, None)


def test_priority_and_continuous_admission():
    blocks = FakeBlocks()
    scheduler = ContinuousBatchingScheduler(2, blocks)
    scheduler.add_request(Sequence("low", [1], 2, priority=0))
    scheduler.add_request(Sequence("high", [1], 2, priority=10))
    running = scheduler.schedule()
    assert [s.sequence_id for s in running] == ["high", "low"]


def test_finished_sequence_is_removed_and_replaced():
    blocks = FakeBlocks()
    scheduler = ContinuousBatchingScheduler(1, blocks)
    scheduler.add_request(Sequence("a", [1], 1))
    scheduler.add_request(Sequence("b", [1], 1))
    assert scheduler.schedule()[0].sequence_id == "a"
    running = scheduler.step({"a": 42})
    assert running[0].sequence_id == "b"
    assert scheduler.running["b"].generated_tokens == []


def test_duplicate_ids_are_rejected():
    blocks = FakeBlocks()
    scheduler = ContinuousBatchingScheduler(1, blocks)
    scheduler.add_request(Sequence("x", [1], 1))
    try:
        scheduler.add_request(Sequence("x", [2], 1))
    except ValueError:
        return
    raise AssertionError("duplicate sequence id was accepted")
