from engine.block_manager import BlockManager
from engine.scheduler import ContinuousBatchingScheduler, Sequence

def test_continuous_admission_and_finish():
    m=BlockManager(8,2); s=ContinuousBatchingScheduler(2,m)
    s.add_request(Sequence("a",[1,2],1)); s.add_request(Sequence("b",[1,2],2)); s.add_request(Sequence("c",[1,2],1))
    assert {x.sequence_id for x in s.schedule()}=={"a","b"}; s.step({"a":3}); assert {x.sequence_id for x in s.running.values()}=={"b","c"}

def test_priority_admission():
    m=BlockManager(8,2); s=ContinuousBatchingScheduler(1,m)
    s.add_request(Sequence("low",[1],1,priority=0)); s.add_request(Sequence("high",[1],1,priority=10)); assert s.schedule()[0].sequence_id=="high"
