#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 26, 2021

from typing import List, Union, Iterable, Dict
from numbers import Number
import numpy as np
from shapely.geometry import LineString
import networkx as nx

from .line_functions import (
    create_line, find_intersects, find_line_intersects, add_points_to_line
)
from .units import get_units


def create_NWN(
    wire_length: float = (7.0 / 7),
    size: Union[tuple, float] = (50.0 / 7),
    density: float = (0.3 * 7**2), 
    seed: int = None,
    conductance: float = (0.1 / 0.1),
    capacitance: float = 1000,
    diameter: float = (50.0 / 50.0),
    resistivity: float = (22.6 / 22.6),
    units: Dict[str, float] = None
) -> nx.Graph:
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
        w = 0.0,
        tau = 0.0,
        epsilon = 0.0,
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
    Converts a JDA NWN to an MNR NWN in-place.

    Parameters
    ----------
    NWN : nx.Graph
        JDA nanowire network.

    """
    if NWN.graph["type"] == "MNR":
        print("Nanowire network already MNR.")
        return

    NWN.graph["type"] = "MNR"
    l0 = NWN.graph["units"]["l0"]
    rho0 = NWN.graph["units"]["rho0"]
    Ron = NWN.graph["units"]["Ron"]
    D0 = NWN.graph["units"]["D0"]
    A0 = D0*D0

    rho = NWN.graph["wire_resistivity"]
    D = NWN.graph["wire_diameter"]
    A = np.pi/4 * D*D

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
        for ind, next_ind in zip(ordering, ordering[1:]):
            # Find inner-wire resistance
            L = NWN.nodes[(i, ind)]["loc"].distance(NWN.nodes[(i, next_ind)]["loc"])
            wire_conductance = (Ron * A0 * A) / (rho0 * l0 * rho * L * 1e3)

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
    NWN: nx.Graph, 
    lines: List[LineString], 
    electrodes: List[bool], 
    resistance: float = None
) -> List[tuple]:
    """
    Adds wires to a given nanowire network in-place. Returns the nodes of the 
    added electrodes in order.

    Currently, adding a wire that already exists breaks things.

    Also, adding wires to a MNR NWN does not work yet.

    Parameters
    ----------
    NWN : Graph
        Nanowire network to add wires to.

    lines : list of LineStrings
        A list of Shapely LineStrings, representing nanowires, to be added.

    electrodes : list of bool
        A list of boolean values specifying whether or not the corresponding
        nanowire in `lines` is an electrode.

    resistance : float, optional
        Junction resistances of the added wires.

    Returns
    -------
    new_wire_nodes : list of tuples
        List of the newly added nodes in the same order in `lines`.
    
    """
    if NWN.graph["type"] != "JDA":
        raise NotImplementedError("Only JDA is currently supported")

    new_wire_num = len(lines)

    if new_wire_num != len(electrodes):
        raise ValueError("Length of new lines list must equal length of electrode boolean list.")

    # Update wire number in NWN
    start_ind = NWN.graph["wire_num"]
    NWN.graph["wire_num"] += new_wire_num

    # Keep track of new wires
    new_wire_nodes = []

    # Add wires to NWN
    for i in range(new_wire_num):
        # Create new node
        NWN.graph["lines"].append(lines[i])
        new_wire_nodes.append((start_ind + i,))
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
            tau = 0.0,
            epsilon = 0.0,
            type = "junction"
        )
        NWN.graph["loc"].update(intersect_dict)

        # Update index lookup
        NWN.graph["node_indices"].update({(start_ind + i,): start_ind + i})

    # Update wire density
    NWN.graph["wire_density"] = (NWN.graph["wire_num"] - len(NWN.graph["electrode_list"])) / NWN.graph["size"]

    return new_wire_nodes


def add_electrodes(NWN: nx.Graph, *args)  -> List[tuple]:
    """
    Convenience function for adding electrodes on the edges of a network 
    in-place. Returns the nodes of the added electrodes in order.

    Can be called in two ways:

        add_electrodes(NWN, *str)
            where *str are strings with values {"left", "right", "top, "bottom"}

        add_electrodes(NWN, *iterable)
            where *iterable are iterables where first entry is a string (as
            before) the second entry is number of electrodes on that side,
            and the third entry is the spacing between the electrodes,
            assumed to be in units of l0. An optional fourth entry can be
            given which is a list of offsets: a float value for each electrode.

    Parameters
    ----------
    NWN : Graph
        Nanowire network.

    *args
        See function description.

    Returns
    -------
    new_wire_nodes : list of tuples
        List of the newly added nodes. If strings were passed, the list
        follows the order passed. If iterables were passed, the list is
        ordered from left-to-right or bottom-to-top concatented. 

    """
    length = NWN.graph["length"]
    width = NWN.graph["width"]
    line_list = []
    seen = []

    # Method one
    if all(isinstance(arg, str) for arg in args):
        for side in args:
            if side in seen:
                raise ValueError(f"Duplicate side: {side}")
            elif side == "left":
                line_list.append(LineString([(0, 0), (0, width)]))
            elif side == "right":
                line_list.append(LineString([(length, 0), (length, width)]))
            elif side == "top":
                line_list.append(LineString([(0, width), (length, width)]))
            elif side == "bottom":
                line_list.append(LineString([(0, 0), (length, 0)]))
            else:
                raise ValueError(f"Invalid side: {side}")
            seen.append(side)

    # Method two
    elif all(isinstance(arg, Iterable) for arg in args):
        for itr in args:
            # Unpack list
            if len(itr) == 3:
                side, num, spacing = itr
                offsets = [0.0] * num
            elif len(itr) == 4:
                side, num, spacing, offsets = itr
            else:
                raise ValueError("Invalid arguments.")

            # Cannot have the same side twice
            if side in seen:
                raise ValueError(f"Duplicate side: {side}")

            # Add electrodes
            elif side == "left":
                for i in range(num):
                    delta = offsets[i]
                    start = (i / num * width) + (spacing / 2)
                    end = ((i + 1) / num * width) - (spacing / 2)
                    line_list.append(LineString(
                        [(0, start + delta), (0, end + delta)]))
            elif side == "right":
                for i in range(num):
                    delta = offsets[i]
                    start = (i / num * width) + (spacing / 2)
                    end = ((i + 1) / num * width) - (spacing / 2)
                    line_list.append(LineString(
                        [(length, start + delta), (length, end + delta)]))
            elif side == "top":
                for i in range(num):
                    delta = offsets[i]
                    start = (i / num * length) + (spacing / 2)
                    end = ((i + 1) / num * length) - (spacing / 2)
                    line_list.append(LineString(
                        [(start + delta, width), (end + delta, width)]))
            elif side == "bottom":
                for i in range(num):
                    delta = offsets[i]
                    start = (i / num * length) + (spacing / 2)
                    end = ((i + 1) / num * length) - (spacing / 2)
                    line_list.append(LineString(
                        [(start + delta, 0), (end + delta, 0)]))
      
    else:
        raise ValueError("Arguments after NWN must be all strings or all lists.")
        
    # Add wires to the network
    new_wire_nodes = add_wires(NWN, line_list, [True] * len(line_list))
    return new_wire_nodes
