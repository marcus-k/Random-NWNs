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


def _capacitance_matrix_JDA(NWN: nx.Graph, drain_node: tuple):
    """
    Create the (sparse) conductance matrix for a given JDA NWN.

    """
    wire_num = NWN.graph["wire_num"]
    M_C = scipy.sparse.dok_matrix((wire_num, wire_num))
    
    for i in range(wire_num):
        for j in range(wire_num):
            if i == j:
                if i == drain_node:
                    M_C[i, j] = 1.0
                else:
                    M_C[i, j] = sum(
                        [NWN[edge[0]][edge[1]]["capacitance"] for edge in NWN.edges((i,))]
                    )

                    # Ground every node with a tiny capacitor: 1e-8 uF
                    M_C[i, j] += 1e-8
            else:
                if i != drain_node:
                    edge_data = NWN.get_edge_data((i,), (j,))
                    if edge_data:
                        M_C[i, j] = -edge_data["capacitance"]
    return M_C


def _capacitance_matrix_MNR(NWN: nx.Graph, drain_node: tuple):
    pass


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
            if i == j:
                if i == drain_node:
                    G[i, j] = 1.0
                else:
                    G[i, j] = sum(
                        [1 / NWN[edge[0]][edge[1]]["resistance"] for edge in NWN.edges((i,))]
                    )

                    # Ground every node with a large resistor: 1/1e8 -> 100 MΩ
                    G[i, j] += 1e-8
            else:
                if i != drain_node:
                    edge_data = NWN.get_edge_data((i,), (j,))
                    if edge_data:
                        G[i, j] = -1 / edge_data["resistance"]
    return G


def _conductance_matrix_MNR(NWN: nx.Graph, drain_node: tuple):
    pass


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


def solve_network(NWN: nx.Graph, source_node: tuple, drain_node: tuple, voltage: float) -> np.ndarray:
    """
    Solve for the voltages of each wire in a given NWN.
    The source node will be at the specified voltage and
    the drain node will be grounded.

    Not fixed for MNR yet.
    
    """
    wire_num = NWN.graph["wire_num"]
    G = conductance_matrix(NWN, drain_node)

    B = scipy.sparse.dok_matrix((wire_num, 1))
    B[source_node, 0] = -1

    C = -B.T

    D = None

    A = scipy.sparse.bmat([[G, B], [C, D]])
    z = scipy.sparse.dok_matrix((wire_num + 1, 1))
    z[-1] = voltage

    # SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
    x = scipy.sparse.linalg.spsolve(A.tocsr(), z)
    return x


