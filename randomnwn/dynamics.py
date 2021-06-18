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
    HP group resistance.

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


def deriv(t: float, y: np.ndarray) -> np.ndarray:
    ...


def solve_evolution(
    NWN: nx.graph, 
    t_eval: np.ndarray,
    func: Callable = HP_resistance,
    args: tuple = tuple(),
):
    """
    Solve parameters of the given nanowire network as various points in time.

    """
    t_span = (t_eval[0], t_eval[-1])