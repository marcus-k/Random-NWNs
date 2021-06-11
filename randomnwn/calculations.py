#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to solve nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   June 8, 2021

import numpy as np
import scipy
import networkx as nx
from networkx.linalg import laplacian_matrix

from .nanowires import get_connected_nodes


def create_matrix(
    NWN: nx.Graph,
    value_type: str = "conductance",
    source_node: tuple = None,
    drain_node: tuple = None,
    ground_nodes: bool = False
) -> scipy.sparse.csr_matrix: 
    """
    Create the Laplacian connectivity matrix.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    value_type : {"conductance", "capacitance"}, optional
        Weight to use for the Laplacian matrix.

    source_node : tuple, optional
        Only needed if ground_nodes is True to find which nodes are to be
        grounded.

    drain_node : tuple, optional
        If a drain node is supplied, the row and column corresponding to 
        the drain node are zeros and a one is placed at the row-column
        intersection. Default: None.

    ground_nodes : bool, optional
        If true, a small value is added to the main diagonal to avoid
        singular matrices.

    Returns
    -------
    M : csr_matrix
        Resultant sparse Laplacian matrix

    """
    TYPES = ["conductance", "capacitance"]
    if value_type not in TYPES:
        raise ValueError("Invalid matrix type.")

    if value_type == "conductance":
        value_type = "is_shorted"

    # Get Laplacian matrix
    nodelist = NWN.graph["node_indices"].keys()
    nodelist_len = len(nodelist)
    M = laplacian_matrix(NWN, nodelist=nodelist, weight=value_type)

    # Ground every node with a huge resistor/tiny capacitor.
    if ground_nodes:
        # Get list of node indices which are not not connected to an electrode
        unconnected_indices = list(set(NWN.graph["node_indices"].values()).difference(
            set(NWN.graph["node_indices"][node] for node in 
                get_connected_nodes(NWN, [source_node, drain_node])
            )
        ))

        tmp = np.zeros(nodelist_len)
        tmp[unconnected_indices] = 1e-12

        # Add small value to diagonal, grounding all non-connected nodes
        M += scipy.sparse.dia_matrix(
            (tmp, [0]), 
            shape = (nodelist_len, nodelist_len)
        )

    # Zero the drain node row and column
    if drain_node is not None:
        M = M.tolil()
        drain_index = NWN.graph["node_indices"][drain_node]
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
        raise ValueError("Invalid solver.")

    return x


def _solve_voltage(
    NWN: nx.Graph, 
    voltage: float, 
    source_node: tuple, 
    drain_node: tuple,
    solver: str,
    **kwargs
) -> np.ndarray:
    """
    Solve for voltages at all the nodes for a given supplied voltage.

    """
    # Find node ordering and indexes
    nodelist = NWN.graph["node_indices"].keys()
    nodelist_len = len(nodelist)
    source_index = NWN.graph["node_indices"][source_node]

    ground_nodes = True if solver == "spsolve" else False

    G = -create_matrix(NWN, "conductance", source_node, drain_node, ground_nodes)
    B = scipy.sparse.dok_matrix((nodelist_len, 1)); B[source_index, 0] = 1
    C = B.T
    D = None

    A = scipy.sparse.bmat([[G, B], [C, D]])
    z = scipy.sparse.dok_matrix((nodelist_len + 1, 1))
    z[-1] = voltage

    out = _solver(A, z, solver, **kwargs)
    return np.array(out, copy=False)


def _solve_current(    
    NWN: nx.Graph, 
    current: float, 
    source_node: tuple, 
    drain_node: tuple,
    solver: str,
    **kwargs
) -> np.ndarray:
    """
    Solve for voltages at all the nodes for a given supplied current.

    """
    # Find node ordering and indexes
    nodelist = NWN.graph["node_indices"].keys()
    nodelist_len = len(nodelist)
    source_index = NWN.graph["node_indices"][source_node]

    ground_nodes = True if solver == "spsolve" else False

    G = create_matrix(NWN, "conductance", source_node, drain_node, ground_nodes)
    z = scipy.sparse.dok_matrix((nodelist_len, 1))
    z[source_index] = current

    out = _solver(G, z, solver, **kwargs)
    return np.array(out, copy=False)


def solve_network(
    NWN: nx.Graph, 
    source_node: tuple, 
    drain_node: tuple, 
    input: float,
    type: str = "voltage",
    solver: str = "spsolve",
    **kwargs
) -> np.ndarray:
    """
    Solve for the voltages of each node in a given NWN. 
    The drain node will be grounded. If the type is voltage, 
    the source node will be the input voltage.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    source_node : tuple
        Voltage/current source node.

    drain_node : tuple
        Grounded output node.

    input : float
        Supplied voltage/current.

    type : {"voltage", "current"}, optional
        Input type. Default: "voltage".

    solver: str, optional
        Name of sparse matrix solving algorithm to use.
        Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.

    Returns
    -------
    out : ndarray
        Output array containing the voltages of each node. If the input type
        is voltage, the current is also in this array.
        
    """

    # # Calculate junction capacitances to determine shorted junctions
    # M_C = capacitance_matrix(NWN, drain_node)

    # # Create block matrix to solve network for voltage and charge
    # B = scipy.sparse.dok_matrix((nodelist_len, 1))
    # B[source_index, 0] = -1

    # C = -B.T

    # D = None

    # A = scipy.sparse.bmat([[M_C, B], [C, D]])
    # z = scipy.sparse.dok_matrix((nodelist_len + 1, 1))
    # z[-1] = voltage

    # # Solve linear equations
    # *voltage_list, charge = scipy.sparse.linalg.spsolve(A.tocsr(), z)


    # # Stored activation data in edges
    # for node1, node2 in NWN.edges():
    #     node1_ind = nodelist.index(node1)
    #     node2_ind = nodelist.index(node2)
    #     voltage_drop = abs(voltage_list[node1_ind] - voltage_list[node2_ind])
        
    #     if voltage_drop < NWN.graph["break_voltage"]:
    #         NWN.edges[(node1, node2)]["is_shorted"] = 0


    # Solve the network
    if type == "voltage":
        out = _solve_voltage(NWN, input, source_node, drain_node, solver, **kwargs)
    elif type == "current":
        out = _solve_current(NWN, input, source_node, drain_node, solver, **kwargs)
    else:
        raise ValueError("Invalid source type.")

    return out


