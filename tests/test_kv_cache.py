import torch
from engine.kv_cache import KVCache

def test_fp_cache_round_trip():
    cache=KVCache(2,4,2,8,dtype=torch.float32,device="cpu")
    k=torch.randn(2,8); v=torch.randn(2,8); cache.store(0,1,k,v); gotk,gotv=cache.get(0,1); assert torch.allclose(gotk,k); assert torch.allclose(gotv,v)

def test_int8_cache_round_trip_is_bounded():
    cache=KVCache(1,2,1,4,dtype=torch.int8,device="cpu")
    k=torch.tensor([[1.,-2.,3.,-4.]]); cache.store(0,0,k,k); got,_=cache.get(0,0); assert torch.allclose(got,k,atol=0.1)
