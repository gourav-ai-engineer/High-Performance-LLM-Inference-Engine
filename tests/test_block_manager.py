import pytest
from engine.block_manager import BlockManager

def test_allocate_and_free():
    m=BlockManager(4,2); assert m.allocate("a",3)==[0,1]; assert m.get_num_free_blocks()==2; m.free("a"); assert m.get_num_free_blocks()==4

def test_cow():
    m=BlockManager(4,2); m.allocate("a",2); m.share("a","b"); new=m.copy_on_write("b",0); assert new != 0; assert m.get_blocks("b")[0]==new; assert m.get_blocks("a")[0]==0

def test_exhaustion():
    m=BlockManager(1,2); m.allocate("a",2)
    with pytest.raises(MemoryError): m.allocate("b",1)
