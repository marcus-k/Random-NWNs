import pytest
from randomnwn import *
from networkx import single_source_dijkstra

@pytest.fixture
def NWN():
    out = create_NWN(
        wire_length = 7.0,
        size = 50.0,
        density = 0.3,
        seed = 123
    )
    return out

def test_NWN_type(NWN):
    assert NWN.graph["type"] == "JDA"

def test_shortest_path(NWN):
    path_len, path = single_source_dijkstra(NWN, (33,), (138,))
    ans = [(33,), (330,), (373,), (622,), (420,), (76,), (21,), (723,), (19,), (232,), (123,), (422,), (308,), (166,), (406,), (53,), (736,), (138,)]

    assert path_len == len(ans) - 1
    assert path == ans

