#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to evolve nanowire networks over time.
# 
# Author: Marcus Kasdorf
# Date:   July 28, 2021

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from numbers import Number
from typing import Callable, List, Union, Tuple, Iterable
from scipy.integrate._ivp.ivp import OdeResult

from .calculations import solve_drain_current
from ._models import (
    resist_func,
    _HP_model_no_decay,
    _HP_model_decay,
    _HP_model_chen,
)


def solve_evolution(
    NWN: nx.Graph, 
    t_eval: np.ndarray,
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]], 
    voltage_func: Callable,
    window_func: Callable = None,
    tol: float = 1e-12,
    model: str = "default",
    solver: str = "spsolve",
    **kwargs
) -> Tuple[OdeResult, List[Tuple]]:
    """
    Solve for the state variables `w` of the junctions of the given nanowire
    network at various points in time with an applied voltage.
    
    Parameters
    ----------
    NWN: Graph
        Nanowire network.

    t_eval : ndarray
        Time points to evaluate the nanowire network at. These should have
        units of `t0`.

    source_node : tuple, or list of tuples
        Voltage source nodes.

    drain_node : tuple, or list of tuples
        Grounded output nodes.

    voltage_func : Callable
        The applied voltage with the calling signature `func(t)`. The voltage 
        should have units of `v0`.

    window_func : Callable, optional
        The window function used in the derivative of `w`. The calling
        signature is `f(w)` where w is the array of state variables.
        The default window function is `f(w) = 1`.

    tol : float, optional
        Tolerance of `scipy.integrate.solve_ivp`. Defaults to 1e-12.

    model : {"default", "decay", "chen"}, optional
        Evolutionary model type. Default: "default".

    solver : str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.
    
    Returns
    -------
    sol : OdeResult
        Output from `scipy.intergrate.solve_ivp`. See the SciPy documentation
        for information on this output's formatting.

    edge_list : list of tuples
        List of the edges corresponding with each `w`.

    """
    if model not in ("default", "decay", "chen"):
        raise ValueError(f'Unsupported model type: model="{model}"')

    # Default window function
    if window_func is None:
        window_func = lambda x: 1

    # Get list of junction edges and state variable w
    edge_list, w0 = map(list, zip(*[
        ((u, v), w) for u, v, w in NWN.edges.data("w") if w is not None]
    ))
    
    # Get list of tau
    tau0 = [tau for _, _, tau in NWN.edges.data("tau") if tau is not None]

    # Get list of epsilon
    epsilon0 = [ep for _, _, ep in NWN.edges.data("epsilon") if ep is not None]

    # Define initial state and associated derivative
    if model == "default":
        _deriv = _HP_model_no_decay
        y0 = w0
    elif model == "decay":
        _deriv = _HP_model_decay
        y0 = w0
    elif model == "chen":
        _deriv = _HP_model_chen
        y0 = np.hstack((w0, tau0, epsilon0))
        if "sigma" not in NWN.graph.keys():
            raise AttributeError(
                "sigma, theta, and a must be set before using the Chen model.")

    # Solve the system of ODEs
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(
        _deriv, t_span, y0, "DOP853", t_eval, 
        atol = tol, 
        rtol = tol,
        args = (NWN, source_node, drain_node, voltage_func, edge_list, 
            window_func, solver, kwargs)
    )

    # Update NWN to final state variables.
    if model == "default":
        set_state_variables(NWN, sol.y[:, -1], edge_list)
    elif model == "decay":
        set_state_variables(NWN, sol.y[:, -1], edge_list)
    elif model == "chen":
        w_list, tau_list, eps_list = np.split(sol.y, 3)
        set_state_variables(NWN, w_list[:, -1], 
            tau_list[:, -1], eps_list[:, -1], edge_list)

    return sol, edge_list


def set_state_variables(NWN: nx.Graph, *args):
    """
    Sets the given nanowire network's state variable. Can be called in the
    following ways:

        set_state_variables(NWN, w)
            where `w` is a scalar value which is set for all edges.

        set_state_variables(NWN, w, edge_list)
            where `w` is an ndarray and edge_list is a list. The `w` array
            contains the state variable for the corresponding edge in 
            `edge_list`.

        set_state_variables(NWN, w, tau)
            where `w` and `tau` are scalar values which is set for all edges.

        set_state_variables(NWN, w, tau, edge_list)
            where `w` and `tau` are ndarrays and edge_list is a list. The `w` 
            and `tau` arrays contains the state variables for the 
            corresponding edge in `edge_list`.

        set_state_variables(NWN, w, tau, epsilon)
            where `w`, `tau`, and `epsilon` are scalar values which is set for 
            all edges.

        set_state_variables(NWN, w, tau, epsilon, edge_list)
            where `w`,`tau`, and `epsilon` are ndarrays and edge_list is a list. 
            The `w`,`tau`, and `epsilon` arrays contains the state variables 
            for the corresponding edge in `edge_list`.

    Parameters
    ----------
    NWN: Graph
        Nanowire network. 

    *args
        See above.
    
    """
    # Only scalar w passed
    if len(args) == 1:
        if isinstance(args[0], Number):
            w = args[0]
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w, "conductance": 1 / R
                } for edge in NWN.edges if NWN.edges[edge]["type"] == "junction"
            }
            nx.set_edge_attributes(NWN, attrs)

        else:
            raise ValueError("Invalid arguments.")

    elif len(args) == 2:
        # vector w and edge_list passed
        if isinstance(args[0], np.ndarray) and isinstance(args[1], Iterable):
            w, edge_list = args
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w[i], "conductance": 1 / R[i]
                } for i, edge in enumerate(edge_list)
            }
            nx.set_edge_attributes(NWN, attrs)

        # scalar w and scalar tau passed
        elif isinstance(args[0], Number) and isinstance(args[1], Number):
            w, tau = args
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w, "conductance": 1 / R, "tau": tau
                } for edge in NWN.edges if NWN.edges[edge]["type"] == "junction"
            }
            nx.set_edge_attributes(NWN, attrs)
            NWN.graph["tau"] = tau

        else:
            raise ValueError("Invalid arguments.")

    elif len(args) == 3:
        # vector w, vector tau, and edge_list passed
        if (isinstance(args[0], np.ndarray) and 
            isinstance(args[1], np.ndarray) and 
            isinstance(args[2], Iterable)):
            w, tau, edge_list = args
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w[i], "conductance": 1 / R[i], "tau": tau[i]
                } for i, edge in enumerate(edge_list)
            }
            nx.set_edge_attributes(NWN, attrs)

        # scalar w, scalar tau, scalar epsilon passed
        if (isinstance(args[0], Number) and 
            isinstance(args[1], Number) and 
            isinstance(args[2], Number)):
            w, tau, epsilon = args
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w, "conductance": 1 / R,
                    "tau": tau, "epsilon": epsilon
                } for edge in NWN.edges if NWN.edges[edge]["type"] == "junction"
            }
            nx.set_edge_attributes(NWN, attrs)
            NWN.graph["tau"] = tau
            NWN.graph["epsilon"] = epsilon

        else:
            raise ValueError("Invalid arguments.")

    elif len(args) == 4:
        # vector w, vector tau, vector epsilon, and edge_list passed
        if (isinstance(args[0], np.ndarray) and 
            isinstance(args[1], np.ndarray) and
            isinstance(args[2], np.ndarray) and
            isinstance(args[3], Iterable)):
            w, tau, epsilon, edge_list = args
            R = resist_func(NWN, w)

            attrs = {
                edge: {
                    "w": w[i], "conductance": 1 / R[i], 
                    "tau": tau[i], "epsilon": epsilon[i]
                } for i, edge in enumerate(edge_list)
            }
            nx.set_edge_attributes(NWN, attrs)

        else:
            raise ValueError("Invalid arguments.")

    else:
        raise ValueError("Invalid number of arguments.")


def get_evolution_current(
    NWN: nx.Graph, 
    sol: OdeResult, 
    edge_list: List[Tuple], 
    source_node: Union[Tuple, List[Tuple]], 
    drain_node: Union[Tuple, List[Tuple]], 
    voltage_func: Callable,
    scaled: bool = False,
    solver: str = "spsolve",
    **kwargs
) -> np.ndarray:
    """
    To be used in conjunction with `solve_evolution`. Takes the output from
    `solve_evolution` and finds the current passing through each drain node
    at each time step.

    The appropriate parameters passed should be the same as `solve_evolution`.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    sol : OdeResult
        Output from `solve_evolution`.

    edge_list : list of tuples
        Output from `solve_evolution`.

    source_node : tuple, or list of tuples
        Voltage source nodes.

    drain_node : tuple, or list of tuples
        Grounded output nodes.

    voltage_func : Callable
        The applied voltage with the calling signature `func(t)`. The voltage 
        should have units of `v0`.

    scaled : bool, optional
        Scale the output by the characteristic values. Default: False.

    solver : str, optional
        Name of sparse matrix solving algorithm to use. Default: "spsolve".

    **kwargs
        Keyword arguments passed to the solver.

    Returns
    -------
    current_array: ndarray
        Array containing the current flow through each drain node. Each column
        corresponds to a drain node in the order passed.

    """
    # Get lists of source and drain nodes
    if isinstance(source_node, tuple):
        source_node = [source_node]
    if isinstance(drain_node, tuple):
        drain_node = [drain_node]
        
    # Preallocate output
    current_array = np.zeros((len(sol.t), len(drain_node)))

    # Loop through each time step
    for i in range(len(sol.t)):
        # Set state variables and get drain currents
        input_V = voltage_func(sol.t[i])
        set_state_variables(NWN, sol.y.T[i], edge_list)
        current_array[i] = solve_drain_current(
            NWN, source_node, drain_node, input_V, scaled, solver, **kwargs)
    
    return current_array.squeeze()
