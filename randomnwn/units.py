#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Characteristic units for a nanowire network.
# 
# Author: Marcus Kasdorf
# Date:   July 1, 2021

from typing import Dict
from networkx import Graph


def derived_units(old_units: Dict):
    """
    Add derived characteristic units to the given dictionary.

    """
    units = old_units.copy()
    units["i0"] = units["v0"] / units["Ron"]                    # A, Current
    units["t0"] = units["D"]**2 / (units["mu0"] * units["v0"])  # μs, Time
    return units


def default_units(with_derived=True):
    """
    Default characteristic units for a nanowire network.

    """
    # Base units
    units = {               # Unit, Description
        "v0": 1.0,          # V, Voltage
        "Ron": 10.0,        # Ω, ON junction resistance
        "l0": 7.0,          # μm, Wire length
        "D0": 50.0,         # nm, Wire diameter
        "D": 10.0,          # nm, Junction length (2x Wire coating thickness)
        "rho0": 22.6,       # nΩm, Wire resistivity
        "mu0": 1e-2,        # μm^2 s^-1 V^-1, Ion mobility
        "Roff_Ron": 160     # none, Off-On Resistance ratio
    }

    # Derived units
    if with_derived:
        units = derived_units(units)

    return units


def set_characteristic_units(
    NWN: Graph, 
    new_units: Dict[str, float] = None
):
    """
    Sets the characteristic units of a nanowire network.

    Parameters
    ----------
    NWN: Graph
        Nanowire network.

    new_units: dict, optional
        Base units to use. If none (default), no units are alter and the
        default units are used.

    """
    if not new_units:
        new_units = dict()

    # Get and update only the base units
    units = default_units(with_derived=False)
    units.update(new_units)

    # Add the derived units
    units = derived_units(units)

    # Add units to nanowire network
    NWN.graph["units"] = units
