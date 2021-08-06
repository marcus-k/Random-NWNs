#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to solve nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 19, 2021

import numpy as np
import scipy
import networkx as nx
from networkx.linalg import laplacian_matrix
from typing import List, Tuple, Set, Union


def get_connected_nodes(NWN: nx.Graph, connected: List[Tuple]) -> Set[Tuple]:
    """
    Returns a list of nodes which are connected to any of the given nodes.

    """
    nodelist = set()
    for subset in nx.connected_components(NWN):
        if any(node in subset for node in connected):
            nodelist = nodelist.union(subset)
    return nodelist


def create_matrix(
    NWN: nx.Graph,
    value_type: str = "conductance",
    source_nodes: List[Tuple] = None,
    drain_nodes: List[Tuple] = None,
    ground_nodes: bool = False
) -> scipy.sparse.csr_matrix: 
    """
    Create the Laplacian connectivity matrix.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    value_type : {"conductance", "capacitance"}, optional
        Weight to use for the Laplacian matrix. Default: "conductance".

    source_nodes : list of tuples, optional
        Only needed if ground_nodes is True to find which nodes are to be
        grounded. Default: None.

    drain_nodes : list of tuples, optional
        If a drain node is supplied, the row and column corresponding to 
        the drain node are zeros and a one is placed at the row-column
        intersection. Default: None.

    ground_nodes : bool, optional
        If true, a small value is added to the main diagonal to avoid
        singular matrices. Default: False.

    Returns
    -------
    M : csr_matrix
        Resultant sparse Laplacian matrix.

    """
    # Error check
    TYPES = ["conductance", "capacitance"]
    if value_type not in TYPES:
        raise ValueError("Invalid matrix type.")

    # Default values
    if source_nodes is None:
        source_nodes = []
    if drain_nodes is None:
        drain_nodes = []

    # Get Laplacian matrix
    nodelist = NWN.graph["node_indices"].keys()
    nodelist_len = len(nodelist)
    M = laplacian_matrix(NWN, nodelist=nodelist, weight=value_type)

    # Ground every node with a huge resistor/tiny capacitor
    if ground_nodes:
        # Get list of node indices which are not connected to an electrode
        unconnected_indices = list(
            set(NWN.graph["node_indices"].values()).difference(
                set(NWN.graph["node_indices"][node] for node in 
                    get_connected_nodes(NWN, [*source_nodes, *drain_nodes])
                )
            )
        )

        small = np.zeros(nodelist_len)
        small[unconnected_indices] = 1e-12

        # Add small value to diagonal, grounding all non-connected nodes
        M += scipy.sparse.dia_matrix(
            (small, [0]), 
            shape = (nodelist_len, nodelist_len)
        )

    # Zero each of the drain nodes' row and column
    for drain in drain_nodes:
        # Change to lil since csr sparsity changes are expensive
        M = M.tolil()
        drain_index = NWN.graph["node_indices"][drain]
        M[drain_index] = 0
        M[:, drain_index] = 0
        M[drain_index, drain_index] = 1

    return M.tocsr()


def _solver(A, z, solver, **kwargs):
    """
    Solve sparse matrix equation.

    """
    if solver == "spsolve":
        x = scipy.sparse.linalg.spsolve(A.tocsr(), z, **kwargs)
    elif solver == "minres":
        x, exit_code = scipy.sparse.linalg.minres(A, z.toarray(), **kwargs)
    elif solver == "lgmres":
        x, exit_code = scipy.sparse.linalg.lgmres(A, z.toarray(), **kwargs)
    elif solver == "gcrotmk":
        x, exit_code = scipy.sparse.linalg.gcrotmk(A, z.toarray(), **kwargs)
    else:
        raise ValueError("Not implemented solver.")
    
    return x


def _solve_voltage(
    NWN: nx.Graph, 
    voltage: float, 
    source_nodes: List[Tuple], 
    drain_nodes: List[Tuple],
    solver: str,
    **kwargs
) -> np.ndarray:
    """
    Solve for voltages at all the nodes for a given supplied voltage.

    """
    # Find node ordering and indexes
    nodelist = NWN.graph["node_indices"].keys()
    source_indices = [NWN.graph["node_indices"][node] for node in source_nodes]

    # Ground nodes only if needed
    ground_nodes = True if solver == "spsolve" else False

    G = -create_matrix(NWN, "conductance", source_nodes, drain_nodes, ground_nodes)
    B = scipy.sparse.dok_matrix((len(nodelist), len(source_indices)))
    for i, ind in enumerate(source_indices):
        B[ind, i] = 1
    C = B.T
    D = None

    A = scipy.sparse.bmat([[G, B], [C, D]])
    z = scipy.sparse.dok_matrix((len(nodelist) + len(source_indices), 1))
    z[len(nodelist):] = voltage

    out = _solver(A, z, solver, **kwargs)
    return np.array(out, copy=False)


def _solve_current(    
    NWN: nx.Graph, 
    current: float, 
    source_nodes: List[Tuple], 
    drain_nodes: List[Tuple],
    solver: str,
    **kwargs
) -> np.ndarray:
    """
    Solve for voltages at all the nodes for a given supplied current.

    """
    # Find node ordering and indexes
    nodelist = NWN.graph["node_indices"].keys()
    source_indices = [NWN.graph["node_indices"][node] for node in source_nodes]

    ground_nodes = True if solver == "spsolve" else False

    G = create_matrix(NWN, "conductance", source_nodes, drain_nodes, ground_nodes)
    z = scipy.sparse.dok_matrix((len(nodelist), 1))
    z[source_indices] = current

    out = _solver(G, z, solver, **kwargs)
    return np.array(out, copy=False)


def solve_network(
    NWN: nx.Graph, 
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]], 
    input: float,
    type: str = "voltage",
    solver: str = "spsolve",
    **kwargs
) -> np.ndarray:
    """
    Solve for the voltages of each node in a given NWN. Each drain node will 
    be grounded. If the type is "voltage", each source node will be at the 
    specified input voltage. If the type is "current", current will be sourced
    from each source node.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    source_node : tuple, or list of tuples
        Voltage/current source nodes.

    drain_node : tuple, or list of tuples
        Grounded output nodes.

    input : float
        Supplied voltage (current) in units of v0 (i0).

    type : {"voltage", "current"}, optional
        Input type. Default: "voltage".

    solver: str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.

    Returns
    -------
    out : ndarray
        Output array containing the voltages of each node. If the input type
        is voltage, the current is also in this array as the last element.
        
    """
    # Get lists of source and drain nodes
    if isinstance(source_node, tuple):
        source_node = [source_node]
    if isinstance(drain_node, tuple):
        drain_node = [drain_node]

    # Pass to solvers
    if type == "voltage":
        out = _solve_voltage(NWN, input, source_node, drain_node, solver, **kwargs)
    elif type == "current":
        out = _solve_current(NWN, input, source_node, drain_node, solver, **kwargs)
    else:
        raise ValueError("Invalid source type.")

    return out


def solve_drain_current(
    NWN: nx.Graph, 
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]], 
    voltage: float,
    scaled: bool = False,
    solver: str = "spsolve",
    **kwargs
) -> np.ndarray:
    """
    Solve for the current through each drain node of a NWN.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    source_node : tuple, or list of tuples
        Voltage source nodes.

    drain_node : tuple, or list of tuples
        Grounded output nodes.

    voltage : float
        Voltage of the source nodes.

    scaled : bool, optional
        Whether or not to scaled the output by i0. Default: False.

    solver: str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    Returns
    -------
    current_array : ndarray
        Array containing the current flow through each drain node in the order
        passed.

    """
    # Get lists of source and drain nodes
    if isinstance(source_node, tuple):
        source_node = [source_node]
    if isinstance(drain_node, tuple):
        drain_node = [drain_node]

    # Preallocate output
    current_array = np.zeros(len(drain_node))

    # Solve nodes
    out = solve_network(
        NWN, source_node, drain_node, voltage, "voltage", solver, **kwargs
    )

    # Find current through each drain node
    for i, drain in enumerate(drain_node):
        I = 0
        for node in NWN.neighbors(drain):
            V = out[NWN.graph["node_indices"][node]]
            R = 1 / NWN.edges[(node, drain)]["conductance"]
            I += V / R
        current_array[i] = I

    # Scale the output if desired
    if scaled:
        current_array *= NWN.graph["units"]["i0"]

    return current_array.squeeze()


def solve_nodal_current(
    NWN: nx.Graph, 
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]], 
    voltage: float,
    scaled: bool = False,
    solver: str = "spsolve",
    **kwargs
) -> np.ndarray:
    """
    Solve for the current through each node of a NWN. It will appear that
    no current is flowing through source (drain) nodes for positive (negative)
    voltages.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    source_node : tuple, or list of tuples
        Voltage source nodes.

    drain_node : tuple, or list of tuples
        Grounded output nodes.

    voltage : float
        Voltage of the source nodes.

    scaled : bool, optional
        Whether or not to scaled the output by i0. Default: False.

    solver: str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    Returns
    -------
    current_array : ndarray
        Array containing the current flow through each drain node in the order
        passed.

    """
    # Get nodal voltages
    V_out = solve_network(
        NWN, source_node, drain_node, voltage, "voltage", solver, **kwargs
    )
    
    # Preallocate output
    current_array = np.zeros(len(NWN.nodes))

    # Calculate input current for each node
    for node in NWN.nodes:
        node_ind = NWN.graph["node_indices"][node]
        for edge in NWN.edges(node):
            edge0_ind = NWN.graph["node_indices"][edge[0]]
            edge1_ind = NWN.graph["node_indices"][edge[1]]
            V_delta = V_out[edge1_ind] - V_out[edge0_ind]

            # Only add current entering a node so we can see how much
            # current passes through. Else, we just get zero due to KCL.
            if V_delta > 0:
                current_array[node_ind] += (V_delta * 
                    NWN.edges[edge]["conductance"] * np.sign(voltage))

    # Scale the output if desired
    if scaled:
        current_array *= NWN.graph["units"]["i0"]

    return current_array


def scale_sol(NWN: nx.Graph, sol: np.ndarray):
    """
    Scale the voltage and current solutions by their characteristic values.

    """
    # Get parameters
    out = np.copy(sol)
    v0 = NWN.graph["units"]["v0"]
    i0 = NWN.graph["units"]["i0"]
    node_num = len(NWN.graph["node_indices"])

    # Current sources
    if node_num == len(sol):
        out *= v0

    # Voltage sources
    else:
        out[:node_num] *= v0
        out[node_num:] *= i0

    return out

