import pytest
from pathlib import Path

from randomnwn import *
from randomnwn.fromtext import *
import networkx as nx
from numpy import hypot
from collections import Counter


@pytest.fixture
def NWN_benchmark_JDA():
    # Get benchmark file
    current_path = Path(__file__).parent.resolve()
    benchmark = current_path.joinpath("test_networks/benchmark.txt")

    # Create nanowire network
    units = {"Ron": 20.0, "rho0": 22.63676, "D0": 60.0, "l0": 1.0}
    NWN = create_NWN_from_txt(str(benchmark), units=units)
    return NWN


@pytest.fixture
def NWN_benchmark_MNR(NWN_benchmark_JDA):
    NWN = NWN_benchmark_JDA
    convert_NWN_to_MNR(NWN)
    return NWN


@pytest.fixture
def NWN_test1():
    NWN = create_NWN(size=(8, 5), seed=123)
    add_electrodes(
        NWN, ["left", 2, 1, [-0.5, 0.5]], ["right", 2, 1, [-0.5, 0.5]]
    )
    return NWN


def test_shortest_path():
    NWN = create_NWN(seed=123)
    assert NWN.graph["type"] == "JDA"

    path_len, path = nx.single_source_dijkstra(NWN, (33,), (138,))
    ans = [(33,), (330,), (373,), (622,), (420,), (76,), (21,), (723,), (19,), 
        (232,), (123,), (422,), (308,), (166,), (406,), (53,), (736,), (138,)]

    assert path_len == len(ans) - 1
    assert path == ans


def test_benchmark_network_JDA(NWN_benchmark_JDA):
    # Get benchmark network
    NWN = NWN_benchmark_JDA
    assert NWN.graph["type"] == "JDA"

    # Calculate JDA resistance
    V = 1.0
    sol = solve_network(NWN, (0,), (1,), V)
    R = V / sol[-1]
    R *= NWN.graph["units"]["Ron"]

    # Check for the correct JDA resistance
    R_JDA = 20 + 20 + 1 / (1 / (20 + 20) + 1 / 20)  # 160/3
    assert abs(R - R_JDA) < 1e-8


def test_benchmark_network_MNR(NWN_benchmark_MNR):
    # Get benchmark network
    NWN = NWN_benchmark_MNR
    assert NWN.graph["type"] == "MNR"
    units = NWN.graph["units"]

    # Calculate MNR resistance
    V = 1.0
    sol = solve_network(NWN, (0,), (1,), V)
    R = V / sol[-1]
    R *= units["Ron"]

    # Check for the correct MNR resistance
    const = units["rho0"] / (np.pi/4 * units["D0"]**2) * 1e3
    Rin1 = const * 1.2
    Rin2 = const * hypot(0.3, 0.3)
    Rin3 = const * hypot(1.5, 1.5)
    Rin4 = const * 1.5

    R_MNR = 20 + Rin1 + 20 + Rin2 + \
        1 / (1 / (Rin3 + 20) + 1 / (20 + Rin4 + 20))    # ~74.618
    assert abs(R - R_MNR) < 1e-8


@pytest.mark.parametrize("NWN", ["NWN_benchmark_JDA", "NWN_test1"])
def test_MNR_node_count(NWN, request):
    NWN = request.getfixturevalue(NWN)
    assert NWN.graph["type"] == "JDA"
    convert_NWN_to_MNR(NWN)

    # Number of wire junction edges
    n_wire_junctions = Counter(nx.get_edge_attributes(NWN, "type").values())["junction"]

    # Number of edges connected to an electrode
    n_edges_electrode = len(NWN.edges(NWN.graph["electrode_list"]))

    # Number of connected electrodes
    n_connected_electrodes = len([node for node, deg in NWN.degree(NWN.graph["electrode_list"]) if deg > 0])

    # Number of isolated wires
    n_isolated_wires = len([x for x in nx.connected_components(NWN) if len(x) == 1])

    # Compare the number of nodes obtained via edge and node count
    node_count = 2 * n_wire_junctions \
        - n_edges_electrode \
        + n_connected_electrodes \
        + n_isolated_wires
    
    assert node_count == NWN.number_of_nodes()
