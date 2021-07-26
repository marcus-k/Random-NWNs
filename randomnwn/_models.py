#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Various dynamic models for nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 26, 2021

import numpy as np
import networkx as nx
from typing import Callable, List, Union, Tuple

from .calculations import solve_network


def resist_func(
    NWN: nx.Graph,
    w: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    The HP group's resistance function in nondimensionalized form.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    w : ndarray or scalar
        Nondimensionalized state variable of the memristor element(s).

    Returns
    -------
    R : ndarray or scalar
        Resistance of the memristor element(s).

    """
    Roff_Ron = NWN.graph["units"]["Roff_Ron"]
    R = w * (1 - Roff_Ron) + Roff_Ron
    return R


def _HP_model_no_decay(
    t: float, 
    w: np.ndarray,
    NWN: nx.Graph,
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]],
    voltage_func: Callable,
    edge_list: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> np.ndarray:
    """
    Derivative of the nondimensionalized state variables `w`.

    """
    if kwargs is None:
        kwargs = dict()

    # Solve for and set resistances
    R = resist_func(NWN, w)
    attrs = {
        edge: {"conductance": 1 / R[i]} for i, edge in enumerate(edge_list)   
    }
    nx.set_edge_attributes(NWN, attrs)

    # Find applied voltage at the current time
    applied_V = voltage_func(t)

    # Solve for voltage at each node
    *V, I = solve_network(
        NWN, source_node, drain_node, applied_V, 
        "voltage", solver, **kwargs
    )
    V = np.array(V)

    # Find voltage differences
    v0, v1 = np.zeros_like(w), np.zeros_like(w)
    for i, edge in enumerate(edge_list):
        v0_indx = NWN.graph["node_indices"][edge[0]]
        v1_indx = NWN.graph["node_indices"][edge[1]]
        v0[i] = V[v0_indx] 
        v1[i] = V[v1_indx]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)
        
    # Find dw/dt
    dwdt = V_delta / R * window_func(w)

    return dwdt


def _HP_model_decay(
    t: float, 
    w: np.ndarray,
    NWN: nx.Graph,
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]],
    voltage_func: Callable,
    edge_list: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> np.ndarray:
    """
    Derivative of the nondimensionalized state variables `w` with 
    decay value `tau`.

    """
    if kwargs is None:
        kwargs = dict()

    # Solve for and set resistances
    R = resist_func(NWN, w)
    attrs = {
        edge: {"conductance": 1 / R[i]} for i, edge in enumerate(edge_list)   
    }
    nx.set_edge_attributes(NWN, attrs)

    # Find applied voltage at the current time
    applied_V = voltage_func(t)

    # Solve for voltage at each node
    *V, I = solve_network(
        NWN, source_node, drain_node, applied_V, 
        "voltage", solver, **kwargs
    )
    V = np.array(V)

    # Find voltage differences
    v0, v1 = np.zeros_like(w), np.zeros_like(w)
    for i, edge in enumerate(edge_list):
        v0_indx = NWN.graph["node_indices"][edge[0]]
        v1_indx = NWN.graph["node_indices"][edge[1]]
        v0[i] = V[v0_indx] 
        v1[i] = V[v1_indx]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)

    # Get decay constant
    tau = NWN.graph["tau"]
        
    # Find dw/dt
    dwdt = (V_delta / R * window_func(w)) - (w / tau)

    return dwdt
