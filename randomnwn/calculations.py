#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to solve nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   May 28, 2021

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
    # Get Laplacian matrix
    wire_num = NWN.graph["wire_num"]
    nodelist = [(i,) for i in range(wire_num)]
    G = laplacian_matrix(NWN, nodelist=nodelist, weight="is_shorted")

    # Ground every node with a huge resistor 
    # to ensure no singular matrices: 1e-8 S -> 100 MΩ
    G += scipy.sparse.dia_matrix(
        (np.ones(wire_num) * 1e-8, [0]), shape=(wire_num, wire_num)
    )

    # Zero the drain node row
    G = G.tolil()
    G[drain_node] = 0
    G[drain_node, drain_node] = 1

    return G


def _conductance_matrix_MNR(NWN: nx.Graph, drain_node: tuple):
    """
    Create the (sparse) conductance matrix for a given MNR NWN.

    """
    # Get Laplacian matrix
    nodelist = sorted(NWN.nodes())
    nodelist_len = len(nodelist)
    G = laplacian_matrix(NWN, nodelist=nodelist, weight="is_shorted")

    # Ground every node with a huge resistor 
    # to ensure no singular matrices: 1e-8 S -> 100 MΩ
    G += scipy.sparse.dia_matrix(
        (np.ones(nodelist_len) * 1e-8, [0]), shape=(nodelist_len, nodelist_len)
    )

    # Zero the drain node row
    G = G.tolil()
    drain_node_index = nodelist.index(drain_node)
    G[drain_node_index] = 0
    G[drain_node_index, drain_node_index] = 1

    return G


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
    
    """
    # Find node ordering and indexes
    nodelist = sorted(NWN.nodes())
    nodelist_len = len(nodelist)
    source_index = nodelist.index(source_node)
    drain_index = nodelist.index(drain_node)


    # Calculate junction capacitances to determine shorted junctions
    M_C = capacitance_matrix(NWN, drain_node)

    # Create block matrix to solve network for voltage and charge
    B = scipy.sparse.dok_matrix((nodelist_len, 1))
    B[source_index, 0] = -1

    C = -B.T

    D = None

    A = scipy.sparse.bmat([[M_C, B], [C, D]])
    z = scipy.sparse.dok_matrix((nodelist_len + 1, 1))
    z[drain_index] = end_voltage
    z[-1] = voltage

    # Solve linear equations
    *voltage_list, charge = scipy.sparse.linalg.spsolve(A.tocsr(), z)


    # Stored activation data in edges
    for node1, node2 in NWN.edges():
        voltage_drop = abs(voltage_list[node1[0]] - voltage_list[node2[0]])
        if voltage_drop < NWN.graph["break_voltage"]:
            NWN.edges[(node1, node2)]["is_shorted"] = 0


    # Solve the network only for the shorted junctions
    G = conductance_matrix(NWN, drain_node)

    B = scipy.sparse.dok_matrix((nodelist_len, 1))
    B[source_index, 0] = -1

    C = -B.T

    D = None

    A = scipy.sparse.bmat([[G, B], [C, D]])
    z = scipy.sparse.dok_matrix((nodelist_len + 1, 1))
    z[drain_index] = end_voltage
    z[-1] = voltage

    # SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
    x = scipy.sparse.linalg.spsolve(A.tocsr(), z)
    return x


