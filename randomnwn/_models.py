#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Various dynamic models for nanowire networks.

# References
# ----------
# [1] https://doi.org/10.1038/nature06932
# [2] https://doi.org/10.1016/j.physleta.2014.08.018
# [3] https://doi.org/10.1088/0957-4484/24/38/384004
# 
# Author: Marcus Kasdorf
# Date:   July 28, 2021

from __future__ import annotations

import numpy as np
import networkx as nx

import numpy.typing as npt
from .typing import *
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from .nanowire_network import NanowireNetwork

from .calculations import solve_network


def resist_func(
    NWN: NanowireNetwork,
    w: float | npt.NDArray
) -> float | npt.NDArray:
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
    w: npt.NDArray,
    NWN: NanowireNetwork,
    source_node: NWNNode | list[NWNNode], 
    drain_node: NWNNode | list[NWNNode],
    voltage_func: Callable,
    edge_list: list,
    start_nodes: list,
    end_nodes: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> npt.NDArray:
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
    v0 = V[start_nodes]
    v1 = V[end_nodes]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)
        
    # Find dw/dt
    dwdt = V_delta / R * window_func(w)

    return dwdt


def _HP_model_decay(
    t: float, 
    w: npt.NDArray,
    NWN: NanowireNetwork,
    source_node: NWNNode | list[NWNNode], 
    drain_node: NWNNode | list[NWNNode],
    voltage_func: Callable,
    edge_list: list,
    start_nodes: list,
    end_nodes: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> npt.NDArray:
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
    v0 = V[start_nodes]
    v1 = V[end_nodes]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)

    # Get decay constant
    tau = NWN.graph["tau"]
        
    # Find dw/dt
    dwdt = (V_delta / R * window_func(w)) - (w / tau)

    return dwdt


def _HP_model_chen(
    t: float, 
    y: npt.NDArray,
    NWN: NanowireNetwork,
    source_node: NWNNode | list[NWNNode], 
    drain_node: NWNNode | list[NWNNode],
    voltage_func: Callable,
    edge_list: list,
    start_nodes: list,
    end_nodes: list,
    window_func: Callable,
    solver: str = "spsolve",
    kwargs: dict = None
) -> npt.NDArray:
    """
    Derivative of the nondimensionalized state variables `w`, `tau`, and
    `epsilon`.

    """
    if kwargs is None:
        kwargs = dict()

    # Unpack values
    w, tau, epsilon = np.split(y, 3)
    sigma, theta, a = NWN.graph["sigma"], NWN.graph["theta"], NWN.graph["a"]

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
    v0 = V[start_nodes]
    v1 = V[end_nodes]
    V_delta = np.abs(v0 - v1) * np.sign(applied_V)
        
    # Find derivatives
    l = V_delta / R
    dw_dt = (l - ((w - epsilon) / tau)) * window_func(w)
    dtau_dt = theta * l * (a - w)
    deps_dt = sigma * l * window_func(w)
    dydt = np.hstack((dw_dt, dtau_dt, deps_dt))

    return dydt


def set_chen_params(NWN: NanowireNetwork, sigma, theta, a):
    NWN.graph["sigma"] = sigma
    NWN.graph["theta"] = theta
    NWN.graph["a"] = a