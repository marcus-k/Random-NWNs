import pytest
from randomnwn import *

def test_NWN_type():
    NWN = create_NWN()
    assert NWN.graph["type"] == "JDA"