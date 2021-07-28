import pytest
from pathlib import Path

from randomnwn import *
from randomnwn.fromtext import *
from networkx import single_source_dijkstra
from numpy import hypot


@pytest.fixture
def NWN():
    out = create_NWN(seed=123)
    return out


def test_NWN_type(NWN):
    assert NWN.graph["type"] == "JDA"


def test_shortest_path(NWN):
    path_len, path = single_source_dijkstra(NWN, (33,), (138,))
    ans = [(33,), (330,), (373,), (622,), (420,), (76,), (21,), (723,), (19,), 
        (232,), (123,), (422,), (308,), (166,), (406,), (53,), (736,), (138,)]

    assert path_len == len(ans) - 1
    assert path == ans


def test_benchmark_network_JDA():
    # Get benchmark file
    current_path = Path(__file__).parent.resolve()
    benchmark = current_path.joinpath("test_networks/benchmark.txt")

    # Create JDA nanowire network
    units = {"Ron": 20.0, "rho0": 22.63676, "D0": 60.0, "l0": 1.0}
    NWN = create_NWN_from_txt(str(benchmark), units=units)

    # Calculate JDA resistance
    V = 1.0
    sol = solve_network(NWN, (0,), (1,), V)
    R = V / sol[-1]
    R *= NWN.graph["units"]["Ron"]

    # Check for the correct JDA resistance
    R_JDA = 20 + 20 + 1 / (1 / (20 + 20) + 1 / 20)  # 160/3
    assert abs(R - R_JDA) < 1e-8


def test_benchmark_network_MNR():
    # Get benchmark file
    current_path = Path(__file__).parent.resolve()
    benchmark = current_path.joinpath("test_networks/benchmark.txt")

    # Create MNR nanowire network
    units = {"Ron": 20.0, "rho0": 22.63676, "D0": 60.0, "l0": 1.0}
    NWN = create_NWN_from_txt(str(benchmark), units=units)
    convert_NWN_to_MNR(NWN)

    # Calculate MNR resistance
    V = 1.0
    sol = solve_network(NWN, (0,), (1,), V)
    R = V / sol[-1]
    R *= NWN.graph["units"]["Ron"]

    # Check for the correct MNR resistance
    const = units["rho0"] / (np.pi/4 * units["D0"]**2) * 1e3
    Rin1 = const * 1.2
    Rin2 = const * hypot(0.3, 0.3)
    Rin3 = const * hypot(1.5, 1.5)
    Rin4 = const * 1.5

    R_MNR = 20 + Rin1 + 20 + Rin2 + \
        1 / (1 / (Rin3 + 20) + 1 / (20 + Rin4 + 20))    # ~74.618
    assert abs(R - R_MNR) < 1e-8