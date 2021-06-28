#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to evolve nanowire networks over time.
# 
# Author: Marcus Kasdorf
# Date:   June 18, 2021

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Callable
from scipy.integrate import solve_ivp

from .calculations import solve_network


def HP_resistance(w: np.ndarray, R_on: float, R_off: float, D: float) -> np.ndarray:
    """
    The HP group's resistance.

    Parameters
    ----------
    w : ndarray
        State variable of the memristor(s).

    Returns
    -------
    R : ndarray
        Resistance of the memristor(s).
    """
    R = (w / D) * (R_on - R_off) + R_off
    return R


def deriv(
    t: float, 
    w: np.ndarray,
    NWN: nx.Graph,
    source_node: tuple, 
    drain_node: tuple,
    voltage_func: Callable,
    edge_list: list,
    solver: str = "spsolve",
    kwargs: dict = None
) -> np.ndarray:
    """
    Derivative of the state variables `w`.

    """
    if kwargs is None:
        kwargs = dict()

    # Get parameters
    R_on = NWN.graph["R_on"]
    R_off = NWN.graph["R_off"]
    mu = NWN.graph["mu"]
    D = 1

    # Solve for resistances
    R = HP_resistance(w, R_on, R_off, D)
    attrs = {
        edge: {"conductance": 1 / R[i]} for i, edge in enumerate(edge_list)
    }
    nx.set_edge_attributes(NWN, attrs)

    # Find applied voltage at the current time
    applied_V = voltage_func(t)

    # Solve for voltages
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
    dwdt = mu * (R_on / D) * V_delta / R

    return dwdt


def solve_evolution(
    NWN: nx.Graph, 
    t_eval: np.ndarray,
    source_node: tuple, 
    drain_node: tuple, 
    voltage_func: Callable,
    solver: str = "spsolve",
    **kwargs
):
    """
    Solve parameters of the given nanowire network as various points in time.

    """
    # Get list of junction edges and the time bounds
    t_span = (t_eval[0], t_eval[-1])
    edge_list, w0 = map(list, zip(*[
        ((u, v), w) for u, v, w in NWN.edges.data("w") if w is not None]
    ))

    # Solve the system of ODEs
    sol = solve_ivp(
        deriv, t_span, w0, "DOP853", t_eval, 
        atol = 1e-12, 
        rtol = 1e-12,
        args = (NWN, source_node, drain_node, voltage_func, edge_list, solver, kwargs)
    )
    final_w = sol.y[:, -1]

    # Update the w value of each edge junction
    attrs = {edge: {"w": final_w[i]} for i, edge in enumerate(edge_list)}
    nx.set_edge_attributes(NWN, attrs)

    return sol, edge_list
