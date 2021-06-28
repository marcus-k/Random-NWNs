#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   June 18, 2021

from typing import List, Union
from numbers import Number
import numpy as np
from shapely.geometry import LineString
import networkx as nx

from .line_functions import *
from .dynamics import *


def create_NWN(
    wire_length: float = 7.0,
    size: Union[tuple, float] = 50.0,
    density: float = 0.3, 
    seed: int = None,
    conductance: float = 0.1,
    capacitance: float = 1000,
    diameter: float = 50.0,
    resistivity: float = 22.6,
    break_voltage: float = -1,
) -> nx.Graph:
    """
    Create a nanowire network stored in a networkx graph. The wires are the 
    graph's vertices, while the wire junctions are represented by the edges.

    The nanowire network started in the junction-dominated assumption, but
    can be converted to the multi-nodal representation after creation. 

    The density might not be attainable with the given size as there can only 
    be a integer number of wires. Thus, the closest density to an integer 
    number of wires is used.

    Parameters
    ----------
    wire_length : float, optional
        Length of each nanowire. Given in micrometers.

    size : tuple or float
        The size of the nanowire network given in micrometers. If a tuple is
        given, it is assumed to be (x-length, y-length). If a number is passed,
        both dimensions will have the same length. The x direction is labeled
        `length`, while the y direction is labeled `width`.

    density : float, optional
        Density of nanowires in the area determined by the width.
        Given in #NW/micrometer^2.

    seed : int, optional
        Seed for random nanowire generation.

    conductance : float, optional
        The junction conductance of the nanowires where they intersect.
        Given in siemens.

    capacitance : float, optional
        The junction capacitance of the nanowires where they intersect.
        Given in microfarads.

    diameter : float, optional
        The diameter of each nanowire. Given in nanometers.

    resistivity : float, optional
        The resistivity of each nanowire. Given in nΩm.
    
    break_voltage : float, optional
        The voltage at which junctions switch from behaving like capacitors
        to resistors. Given in volts.

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

    # Create NWN graph
    NWN = nx.Graph(
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
        break_voltage = break_voltage,
        electrode_list = [],
        lines = [],
        type = "JDA",
        R_on = 10,
        R_off = 1600,
        mu = 1,
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


def convert_NWN_to_MNR(NWN: nx.Graph):
    """
    Converts a NWN in-place from the junction-dominated assumption to the 
    multi-nodal representation.

    Parameters
    ----------
    NWN : nx.Graph
        JDA nanowire network.

    """
    if NWN.graph["type"] == "MNR":
        print("Nanowire network already MNR.")
        return

    NWN.graph["type"] = "MNR"

    for i in range(NWN.graph["wire_num"]):
        # Get the junctions for a wire
        junctions = NWN.edges((i,))

        # If there's only one junction, nothing needs to be changed
        if len(junctions) < 2:
            continue

        # Get location of the junction for a wire
        junction_locs = {
            edge: NWN.graph["loc"][tuple(sorted([edge[0][0], edge[1][0]]))] for edge in junctions
        }

        # Add junctions as part of the LineString that makes up the wire
        NWN.graph["lines"][i], ordering = add_points_to_line(
            NWN.graph["lines"][i], junction_locs.values(), return_ordering=True
        )

        # If wire is electrode, move on to the next wire
        if (i,) in NWN.graph["electrode_list"]:
            continue

        # Split nodes into subnodes representing the junctions on the wires
        for j, (edge, loc) in enumerate(junction_locs.items()):
            # Find connecting node
            other_node = edge[~edge.index((i,))]

            # Get old edge attributes
            old_attributes = NWN.edges[edge]

            # Create the replacing MNR node and edge
            NWN.add_node((i, j), loc=loc, electrode=False)
            NWN.add_edge((i, j), other_node, **old_attributes)

        # Remove old JDA node
        NWN.remove_node((i,))

        # Add edges between subnodes
        D = NWN.graph["wire_diameter"]      # nm
        rho = NWN.graph["wire_resistivity"] # nΩm

        for ind, next_ind in zip(ordering, ordering[1:]):
            # Find inner-wire resistance
            L = NWN.nodes[(i, ind)]["loc"].distance(NWN.nodes[(i, next_ind)]["loc"])
            wire_conductance = (np.pi/4 * D*D) / (rho * L * 1e3)

            # Add inner-wire edge
            NWN.add_edge(
                (i, ind), (i, next_ind), 
                conductance = wire_conductance, 
                capacitance = 0,
                type = "inner"
            )

    # Update index lookup
    NWN.graph["node_indices"] = {
        node: ind for ind, node in enumerate(sorted(NWN.nodes()))
    }


def add_wires(
    NWN: nx.Graph, lines: List[LineString], 
    electrodes: List[bool], 
    resistance: List[float] = None
):
    """
    Adds wires to a given nanowire network.

    Currently, adding a wire that already exists breaks things.

    Also, adding wires to a MNR NWN does not work yet.

    Parameters
    ----------
    NWN : Graph
        Nanowire network to add wires to.

    lines : list of LineStrings
        A list of shapely LineStrings, representing nanowires, to be added.

    electrodes : list of bool
        A list of boolean values specifying whether or not the corresponding
        nanowire in `lines` is an electrode.

    resistance : float
        Junction resistances of the added wires.
    
    """
    if NWN.graph["type"] == "MNR":
        raise NotImplementedError()

    new_wire_num = len(lines)

    if new_wire_num != len(electrodes):
        raise ValueError("Length of new lines list must equal length of electrode boolean list.")

    # Update wire number in NWN
    start_ind = NWN.graph["wire_num"]
    NWN.graph["wire_num"] += new_wire_num

    # Add wires to NWN
    for i in range(new_wire_num):
        NWN.graph["lines"].append(lines[i])
        NWN.add_node(
            (start_ind + i,), 
            electrode = electrodes[i]
        )
        
        # Keep track of the electrodes
        if electrodes[i]:
            NWN.graph["electrode_list"].append((start_ind + i,))

        # Find intersects
        intersect_dict = find_line_intersects(start_ind + i, NWN.graph["lines"])
        
        # Add edges to NWN
        conductance = 1 / resistance if resistance is not None else NWN.graph["junction_conductance"]
        NWN.add_edges_from(
            [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
            conductance = conductance,
            capacitance = NWN.graph["junction_capacitance"],
            w = 0.0,
            type = "junction"
        )
        NWN.graph["loc"].update(intersect_dict)

        # Update index lookup
        NWN.graph["node_indices"].update({(start_ind + i,): start_ind + i})

    # Update wire density
    NWN.graph["wire_density"] = (NWN.graph["wire_num"] - len(NWN.graph["electrode_list"])) / NWN.graph["size"]


# def set_junction_resistance(NWN: nx.Graph, R: float):
#     """
#     Sets the junction resistance for a given nanowire network.
#
#     """
#     NWN.graph["junction_conductance"] = 1 / R
#     attrs = {
#         edge: {
#             "conductance": 1 / R
#         } for edge in NWN.edges() if NWN.edges[edge]["type"] == "junction"
#     }
#     nx.set_edge_attributes(NWN, attrs)
