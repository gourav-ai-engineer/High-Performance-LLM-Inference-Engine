from __future__ import annotations
import argparse, time
from engine.block_manager import BlockManager
from engine.scheduler import ContinuousBatchingScheduler, Sequence

def run(batch_size: int, sequence_length: int):
    manager = BlockManager(max(1024, batch_size * ((sequence_length + 15)//16)), 16)
    scheduler = ContinuousBatchingScheduler(batch_size, manager)
    for i in range(batch_size): scheduler.add_request(Sequence(str(i), [1] * sequence_length, 8))
    started = time.perf_counter(); scheduler.schedule(); elapsed = time.perf_counter() - started
    return elapsed

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--batch-sizes", nargs="+", type=int, default=[1,8,32,128]); p.add_argument("--sequence-lengths", nargs="+", type=int, default=[128,512,2048,4096]); a=p.parse_args()
    for b in a.batch_sizes:
        for s in a.sequence_lengths: print(f"batch={b:3d} seq={s:4d} schedule_ms={run(b,s)*1000:.3f}")
