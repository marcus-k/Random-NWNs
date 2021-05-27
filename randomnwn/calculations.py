#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to solve nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   May 24, 2021

import numpy as np
import scipy
import networkx as nx
from networkx.linalg import laplacian_matrix


def _capacitance_matrix_JDA(NWN: nx.Graph, drain_node: tuple):
    """
    Create the (sparse) conductance matrix for a given JDA NWN.

    """
    # Get Laplacian matrix
    wire_num = NWN.graph["wire_num"]
    nodelist = [(i,) for i in range(wire_num)]
    M_C = laplacian_matrix(NWN, nodelist=nodelist, weight="capacitance")

    # Ground every node with a tiny capacitor 
    # to ensure no singular matrices: 1e-8 uF
    M_C += scipy.sparse.dia_matrix(
        (np.ones(wire_num) * 1e-8, [0]), shape=(wire_num, wire_num)
    )

    # Zero the drain node row
    M_C = M_C.tolil()
    M_C[drain_node] = 0
    M_C[drain_node, drain_node] = 1

    return M_C


def _capacitance_matrix_MNR(NWN: nx.Graph, drain_node: tuple):
    """
    Create the (sparse) conductance matrix for a given MNR NWN.

    """
    # Get Laplacian matrix
    nodelist = sorted(NWN.nodes())
    nodelist_len = len(nodelist)
    M_C = laplacian_matrix(NWN, nodelist=nodelist, weight="capacitance")

    # Ground every node with a tiny capacitor 
    # to ensure no singular matrices: 1e-8 uF
    M_C += scipy.sparse.dia_matrix(
        (np.ones(nodelist_len) * 1e-8, [0]), shape=(nodelist_len, nodelist_len)
    )

    # Zero the drain node row
    M_C = M_C.tolil()
    drain_node_index = nodelist.index(drain_node)
    M_C[drain_node_index] = 0
    M_C[drain_node_index, drain_node_index] = 1

    return M_C


def capacitance_matrix(NWN: nx.Graph, drain_node: tuple):
    """
    Create the capacitance matrix for a given NWN with a specified drain node.

    """
    if NWN.graph["type"] == "JDA":
        return _capacitance_matrix_JDA(NWN, drain_node[0])
    elif NWN.graph["type"] == "MNR":
        return _capacitance_matrix_MNR(NWN, drain_node)
    else:
        raise ValueError("Nanowire network has invalid type.")


def _conductance_matrix_JDA(NWN: nx.Graph, drain_node: int):
    """
    Create the (sparse) conductance matrix for a given JDA NWN.

    """
    wire_num = NWN.graph["wire_num"]
    G = scipy.sparse.dok_matrix((wire_num, wire_num))
    
    for i in range(wire_num):
        for j in range(wire_num):
            # Main diagonal
            if i == j:
                if i == drain_node:
                    G[i, j] = 1.0
                else:
                    G[i, j] = sum(
                        [1 / NWN.edges[edge]["resistance"] for edge in NWN.edges((i,)) if NWN.edges[edge]["is_shorted"]]
                    )

                    # Ground every node with a large resistor: 1/1e8 -> 100 MΩ
                    G[i, j] += 1e-8

            # All non-diagonal elements except the drain node row
            elif i != drain_node:
                edge_data = NWN.get_edge_data((i,), (j,))
                if edge_data is not None and edge_data["is_shorted"]:
                    G[i, j] = -1 / edge_data["resistance"]

    return G


def _conductance_matrix_MNR(NWN: nx.Graph, drain_node: tuple):
    raise NotImplementedError()


def conductance_matrix(NWN: nx.Graph, drain_node: tuple):
    """
    Create the conductance matrix for a given NWN with a specified drain node.

    """
    if NWN.graph["type"] == "JDA":
        return _conductance_matrix_JDA(NWN, drain_node[0])
    elif NWN.graph["type"] == "MNR":
        return _conductance_matrix_MNR(NWN, drain_node)
    else:
        raise ValueError("Nanowire network has invalid type.")


def solve_network(
    NWN: nx.Graph, 
    source_node: tuple, 
    drain_node: tuple, 
    voltage: float,
    end_voltage: float = 0
) -> np.ndarray:
    """
    Solve for the voltages of each wire in a given NWN.
    The source node will be at the specified voltage and
    the drain node will be grounded.

    Not fixed for MNR yet.
    
    """
    # Calculate junction capacitances to determine shorted junctions
    M_C = capacitance_matrix(NWN, drain_node)

    # Create block matrix to solve network for voltage and charge
    wire_num = NWN.graph["wire_num"]
    B = scipy.sparse.dok_matrix((wire_num, 1))
    B[source_node, 0] = -1

    C = -B.T

    D = None

    A = scipy.sparse.bmat([[M_C, B], [C, D]])
    z = scipy.sparse.dok_matrix((wire_num + 1, 1))
    z[drain_node] = end_voltage
    z[-1] = voltage

    # Solve linear equations
    *voltage_list, charge = scipy.sparse.linalg.spsolve(A.tocsr(), z)

    # Stored activation data in edges
    for node1, node2 in NWN.edges():
        voltage_drop = abs(voltage_list[node1[0]] - voltage_list[node2[0]])
        if voltage_drop > NWN.graph["break_voltage"]:
            NWN.edges[(node1, node2)]["is_shorted"] = True
        else:
            NWN.edges[(node1, node2)]["is_shorted"] = False



    # Solve the network only for the shorted junctions
    G = conductance_matrix(NWN, drain_node)

    B = scipy.sparse.dok_matrix((wire_num, 1))
    B[source_node, 0] = -1

    C = -B.T

    D = None

    A = scipy.sparse.bmat([[G, B], [C, D]])
    z = scipy.sparse.dok_matrix((wire_num + 1, 1))
    z[drain_node] = end_voltage
    z[-1] = voltage

    # SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
    x = scipy.sparse.linalg.spsolve(A.tocsr(), z)
    return x


