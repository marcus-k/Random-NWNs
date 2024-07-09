#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 8, 2024

from __future__ import annotations

import networkx as nx
import numpy as np
from collections import Counter

import numpy.typing as npt
from typing import Callable, Literal, TypeAlias, Any
from shapely.geometry import LineString, Point
from numbers import Number
from .typing import *

from .units import get_units
from ._models import resist_func
from .line_functions import create_line, find_intersects
from .nanowires import convert_NWN_to_MNR


class ParameterNotSetError(Exception):
    """
    Raised when an parameter that needs to used has not been set yet.

    Parameters
    ----------
    param : str
        Name of the parameter that needs to be set.

    message : str, optional
        Explanation of the error.

    """
    def __init__(self, message: str, param: Any | None = None):
        super().__init__(message)
        self.param = param


class NanowireNetwork(nx.Graph):
    """
    Internal nanowire network object. Should not be instantiated directly.
    Use `create_NWN()` instead.

    Parameters
    ----------
    incoming_graph_data : input graph, optional
        Data to initialize graph. Same as networkx.Graph object.

    **attr
        Keyword arguments. Same as networkx.Graph object.

    """
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._resist_func = None
        self._state_vars = []
        self._wire_junctions = None

    @property
    def type(self) -> Literal["JDA", "MNR"]:
        return self.graph["type"]

    @property
    def electrodes(self) -> list[JDANode]:
        return self.graph["electrode_list"]

    @property
    def n_electrodes(self) -> int:
        return len(self.electrodes)

    @property
    def n_wires(self) -> int:
        """Number of wires in the network. Does not include electrodes."""
        return self.graph["wire_num"] - self.n_electrodes

    @property
    def n_inner_junctions(self) -> int | None:
        if self.type != "MNR":
            return None

        _ = Counter(nx.get_edge_attributes(self, "type").values())
        return _["inner"]

    @property
    def n_wire_junctions(self) -> int:
        _ = Counter(nx.get_edge_attributes(self, "type").values())
        return _["junction"]

    @property
    def units(self) -> dict[str, float]:
        return self.graph["units"]

    @property
    def lines(self) -> list[LineString]:
        """List of LineStrings representing the nanowires. Includes electrodes."""
        return self.graph["lines"]

    @property
    def wire_density(self) -> float:
        """Wire density in units of (l0)^-2. Does not include electrodes."""
        return self.graph["wire_density"]

    @property
    def loc(self) -> dict[tuple[int, int], Point]:
        """Dictionary of wire junction locations."""
        return self.graph["loc"]

    def get_index(self, node: NWNNode) -> int:
        """Return the unique index of a node in the network."""
        return self.graph["node_indices"][node]

    def get_node(self, index: int) -> NWNNode:
        """Return the node corresponding to the index."""
        try:
            return next(k for k, v in self.graph["node_indices"].items() if v == index)
        except StopIteration as e:
            raise ValueError("given index does not have a node") from e

    def to_MNR(self) -> None:
        convert_NWN_to_MNR(self)

    @property
    def wire_junctions(self) -> list[NWNEdge]:
        """
        Return a list of edges with the "type" attribute set to "junction". 
        Once called, the list is cached. If wires are added, clear the cache
        by deleting the property.

        """
        if self._wire_junctions is None:
            self._wire_junctions = [(u, v) for u, v, d in self.edges(data=True) if d["type"] == "junction"]

        return self._wire_junctions
    
    @wire_junctions.deleter
    def wire_junctions(self) -> None:
        self._wire_junctions = None
    
    @property
    def state_vars(self) -> list[str]:
        return self._state_vars
    
    @state_vars.setter
    def state_vars(self, names: list[str]) -> None:
        self._state_vars = names

    @property
    def resistance_function(
        self
    ) -> Callable[[NanowireNetwork, npt.ArrayLike], npt.ArrayLike]:
        """
        Resistance function of the nanowire network. Should have the calling
        signature `func(NWN, param1, [param2, ...])` where `NWN` is the
        nanowire network, and `param1`, `param2`, etc. are the state variables
        of the memristor element(s).

        You can also pass the string "linear" to choose a default linear
        resistance function based on the state variable w in [0, 1].

        """
        return self._resist_func

    @resistance_function.setter
    def resistance_function(
        self, 
        func: str | Callable[[NanowireNetwork, npt.ArrayLike], npt.ArrayLike]
    ) -> None:
        if func == "linear":
            self._resist_func = resist_func
        else:
            self._resist_func = func

    def set_state_var(
        self,
        var_name: str,
        value: npt.ArrayLike,
        edge_list: list[NWNEdge] | None = None,
    ) -> None:
        """
        Set the state variable of the memristors (nanowire network wire 
        junctions) in the network.

        Parameters
        ----------
        var_name : str
            Name of the state variable(s) to set.

        value : ndarray or scalar
            Value to set the state variable(s) to.

        """
        value = np.atleast_1d(value)

        if var_name not in self.state_vars:
            cls = self.__class__
            raise ParameterNotSetError(
                f"'{var_name}' is not in {cls.__qualname__}.state_vars (currently is {self.state_vars})."
                f"\nDid you set it using {cls.__qualname__}.state_vars = ['{var_name}', ...]?"
            )

        # Set the state variable for all edges
        if edge_list is None and value.size == 1:
            nx.set_edge_attributes(self, {
                edge: {var_name: value[0]} for edge in self.wire_junctions
            })

        # Set the state variable for a subset of edges
        elif len(edge_list) == value.size:
            nx.set_edge_attributes(self, {
                edge: {var_name: value[i]} for i, edge in enumerate(edge_list)
            })

        else:
            raise ValueError("Length of edge_list does not match the length of the value array.")
        
    def get_state_var(self, var_name: str, edge_list: list[NWNEdge] | None = None) -> npt.ArrayLike:
        """
        Get the state variable of the memristors (nanowire network wire 
        junctions) in the network.

        Parameters
        ----------
        var_name : str
            Name of the state variable(s) to get.

        edge_list : list of edges, optional
            List of edges to get the state variable from. If None, all edges
            will be used.

        Returns
        -------
        ndarray or scalar
            Value of the state variable(s).

        """
        if var_name not in self.state_vars:
            cls = self.__class__
            raise AttributeError(
                f"'{var_name}' is not in {cls.__qualname__}.state_vars (currently is {self.state_vars})."
                f"\nDid you set it using {cls.__qualname__}.state_vars = ['{var_name}', ...]?"
            )

        if edge_list is None:
            edge_list = self.wire_junctions

        try:
            return np.array([self[edge[0]][edge[1]][var_name] for edge in edge_list])
        except KeyError as e:
            raise ParameterNotSetError(f"'{var_name}' has not been set yet using `set_state_var`.") from e

    def update_resistance(
        self, 
        state_var_vals: npt.ArrayLike | list[npt.ArrayLike], 
        edge_list: list[NWNEdge]
    ) -> None:
        """
        Update the resistance of the nanowire network based on the provided
        state variable values. The resistance function should be set before 
        calling this method.

        Parameters
        ----------
        state_var_vals : ndarray or list of ndarrays
            Values of the state variables to use in the resistance function.
            The should be in the same order as the state variables. If the
            resistance function takes multiple state variables, pass a list of
            arrays in the same order as `NanowireNetwork.state_vars`.

        edge_list : list of edges
            List of edges to update the resistance for.

        """
        if self.resistance_function is None:
            raise ParameterNotSetError("Resistance function attribute must be set before updating resistance.")

        if not isinstance(state_var_vals, list):
            state_var_vals = [state_var_vals]

        R = self.resistance_function(self, *state_var_vals)
        attrs = {
            edge: {"conductance": 1 / R[i]} for i, edge in enumerate(edge_list)   
        }
        nx.set_edge_attributes(self, attrs)
        

    def __repr__(self) -> str:
        d = {
            "Type": self.type,
            "Wires": self.n_wires,
            "Electrodes": self.n_electrodes,
            "Inner-wire junctions": self.n_inner_junctions,
            "Wire junctions": self.n_wire_junctions,
            "Length": f"{self.graph['length'] * self.units['l0']:#.4g} um ({self.graph['length']:#.4g} l0)",
            "Width": f"{self.graph['width'] * self.units['l0']:#.4g} um ({self.graph['width']:#.4g} l0)",
            "Wire Density": f"{self.graph['wire_density'] / self.units['l0']**2:#.4g} um^-2 ({self.graph['wire_density']:#.4g} l0^-2)"
        }
        # Get max key length
        m = max(map(len, list(d.keys())))

        # Create string representation
        s = "\n".join([f"{k:>{m}}: {v}" for k, v in d.items()])
        return s


def create_NWN(
    wire_length: float = (7.0 / 7),
    size: tuple | float = (50.0 / 7),
    density: float = (0.3 * 7**2),
    seed: int = None,
    conductance: float = (0.1 / 0.1),
    capacitance: float = 1000,
    diameter: float = (50.0 / 50.0),
    resistivity: float = (22.6 / 22.6),
    units: dict[str, float] = None
) -> NanowireNetwork:
    """
    Create a nanowire network represented by a NetworkX graph. Wires are 
    represented by the graph's vertices, while the wire junctions are 
    represented by the graph's edges.

    The nanowire network starts in the junction-dominated assumption (JDA), but
    can be converted to the multi-nodal representation (MNR) after creation. 

    The density may not be attainable with the given size, as there can only 
    be a integer number of wires. Thus, the closest density to an integer 
    number of wires is used.

    See `units.py` for the units used by the parameters. 

    Parameters
    ----------
    wire_length : float, optional
        Length of each nanowire. Given in units of l0.

    size : 2-tuple or float
        The size of the nanowire network given in units of l0. If a tuple is
        given, it is assumed to be (x-length, y-length). If a number is passed,
        both dimensions will have the same length. The x direction is labeled
        `length`, while the y direction is labeled `width`.

    density : float, optional
        Density of nanowires in the area determined by the width.
        Given in units of (l0)^-2.

    seed : int, optional
        Seed for random nanowire generation.

    conductance : float, optional
        The junction conductance of the nanowires where they intersect.
        Given in units of (Ron)^-1.

    capacitance : float, optional
        The junction capacitance of the nanowires where they intersect.
        Given in microfarads. (Currently unused)

    diameter : float, optional
        The diameter of each nanowire. Given in units of D0.

    resistivity : float, optional
        The resistivity of each nanowire. Given in units of rho0.

    units : dict, optional
        Dictionary of custom base units. Defaults to None which will use the 
        default units given in `units.py`

    Returns
    -------
    NWN : Graph
        The created random nanowire network.

    """
    # Convert size to length and width, size will be the area
    if isinstance(size, tuple):
        length, width = size
        size = size[0] * size[1]
    elif isinstance(size, Number):
        length, width = size, size
        size = size * size
    else:
        raise ValueError("Invalid size type.")

    # Get closest density with an integer number of wires.
    wire_num = round(size * density)
    density = wire_num / size

    # Get characteristic units
    units = get_units(units)

    # Create NWN graph
    NWN = NanowireNetwork(
        wire_length = wire_length,
        length = length,
        width = width,
        size = size,
        wire_density = density,
        wire_num = wire_num,
        junction_conductance = conductance,
        junction_capacitance = capacitance,
        wire_diameter = diameter,
        wire_resistivity = resistivity,
        electrode_list = [],
        lines = [],
        type = "JDA",
        units = units,
        tau = 0.0,
        epsilon = 0.0,
    )

    # Create seeded random generator for testing
    rng = np.random.default_rng(seed)

    # Add the wires as nodes to the graph
    for i in range(NWN.graph["wire_num"]):
        NWN.graph["lines"].append(create_line(
            NWN.graph["wire_length"],
            xmax = NWN.graph["length"],
            ymax = NWN.graph["width"],
            rng = rng
        ))
        NWN.add_node((i,), electrode=False)

    # Find intersects and create the edges (junctions)
    intersect_dict = find_intersects(NWN.graph["lines"])
    NWN.add_edges_from(
        [((key[0],), (key[1],)) for key in intersect_dict.keys()],
        conductance = conductance,
        capacitance = capacitance,
        type = "junction"
    )
    NWN.graph["loc"] = intersect_dict

    # Find junction density
    NWN.graph["junction_density"] = len(intersect_dict) / size

    # Create index lookup
    NWN.graph["node_indices"] = {
        node: ind for ind, node in enumerate(sorted(NWN.nodes()))
    }

    return NWN