#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Characteristic units for a nanowire network.
# 
# Author: Marcus Kasdorf
# Date:   July 1, 2021

def default_units():
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
    units["i0"] = units["v0"] / units["Ron"]                    # A, Current
    units["t0"] = units["D"]**2 / (units["mu0"] * units["v0"])  # μs, Time

    return units