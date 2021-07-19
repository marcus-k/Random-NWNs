#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks from text file.
# Testing only, not for production.
# 
# Author: Marcus Kasdorf
# Date:   July 15, 2021

from typing import Dict
import numpy as np
from shapely.geometry import LineString
import networkx as nx

from .line_functions import find_intersects
from .units import get_units

def create_NWN_from_txt(
    filename: str,
    conductance: float = 1.0,
    diameter: float = 1.0,
    resistivity: float = 1.0,
    units: Dict[str, float] = None
) -> nx.Graph:
    """
    Create a nanowire network represented by a NetworkX graph. Wires are 
    represented by the graph's vertices, while the wire junctions are 
    represented by the graph's edges.

    The text file input is assumed to be in two columns: x and y.
    Each wire is a pair of rows, one row containing the start point and the
    following containing the end point. The first two pairs of rows are the
    electrodes.

    Parameters
    ----------
    filename : str
        Text file containing the start and end locations of the wires.

    conductance : float, optional
        The junction conductance of the nanowires where they intersect.
        Given in units of (Ron)^-1.

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
    # Get coordinates from text file
    x, y = np.loadtxt(filename, unpack=True)

    # Convert to LineStrings
    line_list = []
    for i in range(0, len(x), 2):
        line_list.append(LineString([(x[i], y[i]), (x[i+1], y[i+1])]))

    # Find dimensions
    length = np.max(x) - np.min(x)
    width = np.max(y) - np.min(y)
    size = length * width

    # Find density
    wire_num = len(line_list)
    density = wire_num / size

    # Get characteristic units
    units = get_units(units)

    # Create NWN graph
    NWN = nx.Graph(
        wire_length = None,
        length = length,
        width = width, 
        size = size,
        wire_density = density, 
        wire_num = wire_num,
        junction_conductance = conductance,
        junction_capacitance = None,
        wire_diameter = diameter,
        wire_resistivity = resistivity,
        electrode_list = [],
        lines = [],
        type = "JDA",
        units = units,
    )

    # Add the wires as nodes to the graph
    for i in range(NWN.graph["wire_num"]):
        NWN.graph["lines"].append(line_list[i])
        if i == 0 or i == 1:
            NWN.add_node((i,), electrode=True)
            NWN.graph["electrode_list"].append((i,))
        else:
            NWN.add_node((i,), electrode=False)
        
    # Find intersects and create the edges (junctions)
    intersect_dict = find_intersects(NWN.graph["lines"])
    NWN.add_edges_from(
        [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
        conductance = conductance,
        capacitance = None,
        w = 0.0,
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
