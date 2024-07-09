#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to modify nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 26, 2021

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .typing import *
from typing import Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    from .nanowire_network import NanowireNetwork

from .line_functions import find_line_intersects, add_points_to_line


def convert_NWN_to_MNR(NWN: NanowireNetwork):
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

        # Remove old JDA node, only if it had connections
        if len(junction_locs) >= 1:
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
    NWN: NanowireNetwork, 
    lines: list[LineString], 
    electrodes: list[bool], 
    resistance: float = None
) -> list[JDANode]:
    """
    Adds wires to a given nanowire network in-place. Returns the nodes of the 
    added wires in order.

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
            type = "junction"
        )
        NWN.graph["loc"].update(intersect_dict)

        # Update index lookup
        NWN.graph["node_indices"].update({(start_ind + i,): start_ind + i})

    # Update wire density
    NWN.graph["wire_density"] = (NWN.graph["wire_num"] - len(NWN.graph["electrode_list"])) / NWN.graph["size"]

    # Clear current junction list
    del NWN.wire_junctions

    return new_wire_nodes


def add_electrodes(NWN: NanowireNetwork, *args)  -> list[JDANode]:
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
        ordered from left-to-right or bottom-to-top concatenated. 

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


def get_edge_indices(NWN: NanowireNetwork, edges: list[NWNEdge]) -> tuple[list[NWNNodeIndex], list[NWNNodeIndex]]:
    """
    Given a NWN and a list of edges, returns two lists: one of the indices of
    the first nodes in the input edge list, and one of the second.

    Parameters
    ----------
    NWN : Graph
        Nanowire Network.

    edges : list of tuples
        List of edges to find the indices of.

    """
    # JDA edge indices
    if NWN.graph["type"] == "JDA":
        start_nodes, end_nodes = map(list, 
            zip(*[(*n1, *n2) for n1, n2 in edges]))

    # MNR edge indices
    elif NWN.graph["type"] == "MNR":
        tmp = []
        for key in NWN.graph["node_indices"].keys():
            if len(key) == 2:
                tmp.append(key[1])
            else:
                tmp.append(0)

        node_start_index = np.where(np.asarray(tmp) == 0)[0]

        start_nodes, end_nodes = [], []
        for n1, n2 in edges:
            if len(n1) == 2:
                start_nodes.append(node_start_index[n1[0]] + n1[1])
            else:
                start_nodes.append(node_start_index[n1[0]])
            if len(n2) == 2:
                end_nodes.append(node_start_index[n2[0]] + n2[1])
            else:
                end_nodes.append(node_start_index[n2[0]])

    # Invalid NWN Type
    else:
        raise ValueError("Invalid NWN type.")

    return start_nodes, end_nodes