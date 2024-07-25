#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Characteristic units for a nanowire network.
# 
# Author: Marcus Kasdorf
# Date:   July 1, 2021


class NWNUnits(dict):
    """
    Class for characteristic units for a nanowire network.

    Parameters
    ----------
    new_units : dict, optional
        Dictionary of any custom units to use. Only base units can be altered.

    Attributes
    ----------
    units : dict
        Dictionary of characteristic units.
    
    """
    def __init__(self, new_units: dict[str, float] = None):
        self.update(get_units(new_units))

        self.settable_units = (
            "v0", "Ron", "l0", "D0", "w0", "rho0", "mu0", "Roff_Ron"
        )
        self.not_settable_units = (
            "i0", "t0"
        )

    def __setitem__(self, key: str, value: float):

        if key in self.settable_units:
            super().__setitem__(key, value)
        elif key in self.not_settable_units:
            raise ValueError(f"Cannot set a derived unit: {key}")
        else:
            raise KeyError(f"Unknown unit {key}.")
            
        # Derived units
        super().__setitem__(
            "i0", self["v0"] / self["Ron"]                      # A, Current
        )      
        super().__setitem__(
            "t0", self["w0"]**2 / (self["mu0"] * self["v0"])    # μs, Time
        )
        
    def __repr__(self) -> str:
        # Get max key length
        m = max(map(len, list(self.keys())))

        # Create string representation
        s = "\n".join([f"{k:>{m}}: {v}" for k, v in self.items()])
        return s
        

def get_units(new_units: dict[str, float] = None) -> dict[str, float]:
    """
    Returns the characteristic units for a nanowire network.

    Parameters
    ----------
    new_units : dict, optional
        Dictionary of any custom units to use. Only base units can be altered.

    Returns
    -------
    units : dict
        Dictionary of characteristic units.
    
    """
    if new_units is None:
        new_units = dict()

    # Base units
    units = {               # Unit, Description
        "v0": 1.0,          # V, Voltage
        "Ron": 10.0,        # Ω, ON junction resistance
        "l0": 7.0,          # μm, Wire length
        "D0": 50.0,         # nm, Wire diameter
        "w0": 10.0,          # nm, Junction length (2x Wire coating thickness)
        "rho0": 22.6,       # nΩm, Wire resistivity
        "mu0": 1e-2,        # μm^2 s^-1 V^-1, Ion mobility
        "Roff_Ron": 160     # none, Off-On Resistance ratio
    }

    # Add any custom units
    units.update(new_units)

    # Derived units
    units["i0"] = units["v0"] / units["Ron"]                    # A, Current
    units["t0"] = units["w0"]**2 / (units["mu0"] * units["v0"])  # μs, Time

    return units

