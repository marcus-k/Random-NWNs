import pytest
from pathlib import Path

from randomnwn import *
from randomnwn.fromtext import *
from randomnwn.nanowire_network import NanowireNetwork


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


@pytest.mark.parametrize("NWN", ["NWN_benchmark_JDA", "NWN_benchmark_MNR", "NWN_test1"])
def test_NWN_edge_indices(NWN, request):
    NWN = request.getfixturevalue(NWN)

    # Get edge indices
    start1, end1 = map(np.asarray, get_edge_indices(NWN, NWN.wire_junctions))
    start2, end2 = np.asarray(NWN.get_index_from_edge(NWN.wire_junctions)).T

    # Check if the edge indices are the same
    assert np.all(start1 == start2)
    assert np.all(end1 == end2)


@pytest.mark.parametrize("NWN_input", ["NWN_benchmark_JDA", "NWN_benchmark_MNR"])
@pytest.mark.parametrize("voltage_func", [
    lambda t: 1 + 0*t,
    lambda t: 2 + 1*t,
])
@pytest.mark.parametrize("window_func", [
    lambda w: 1,
    lambda w: w * (1 - w),
])
def test_new_api_evolution(NWN_input, voltage_func, window_func, request):
    # Common vars
    max_time = 10
    n_time = 100
    t_eval = np.linspace(0, max_time, n_time)

    # Evolution with new API
    NWN: NanowireNetwork = request.getfixturevalue(NWN_input)
    source, drain = NWN.electrodes

    NWN.state_vars = ["w"]
    NWN.set_state_var("w", 0.05)

    sol1 = NWN.evolve(
        "default", t_eval, source, drain, voltage_func,
        window_func, ivp_options={"rtol": 1e-7, "atol": 1e-7}
    )

    # Evolution with old API
    NWN: NanowireNetwork = request.getfixturevalue(NWN_input)
    source, drain = NWN.electrodes

    set_state_variables(NWN, 0.05)

    sol2, _ = solve_evolution(
        NWN, t_eval, source, drain, voltage_func, window_func, 1e-7, "default"
    )

    # Compare the results
    assert np.all(sol1.y == sol2.y)
