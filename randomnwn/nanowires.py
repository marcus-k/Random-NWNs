#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   May 24, 2021

from typing import List
import numpy as np
from shapely.geometry import LineString
import networkx as nx
import matplotlib.pyplot as plt

from .line_functions import *


def create_NWN(
    wire_length: float = 7.0, 
    width: float = 50.0, 
    density: float = 0.3, 
    seed: int = None,
    resistance: float = 10,
    capacitance: float = 1000,
    break_voltage: float = 1
) -> nx.Graph:
    """
    Create a nanowire network stored in a networkx graph. The wires are the 
    graph's vertices, while the wire junctions are represented by the edges.

    The nanowire network started in the junction-dominated assumption, but
    can be converted to the multi-nodal representation after creation. 

    The density might not be attainable with the given size as there can only 
    be a integer number of wires. Thus, the closest density to an integer 
    number of wires is used.

    Wire length and grid width are in micrometers. Resistance is in ohms.
    Capacitance is in microfarads.

    """
    # Get closest density with an integer number of wires.
    size = width * width
    wire_num = round(size * density)
    density = wire_num / size

    # Create NWN graph
    NWN = nx.Graph(
        wire_length = wire_length, 
        width = width, 
        size = size,
        wire_density = density, 
        wire_num = wire_num,
        junction_resistance = resistance,
        junction_capacitance = capacitance,
        break_voltage = break_voltage,
        electrode_list = [],
        lines = [],
        type = "JDA"
    )

    # Create seeded random generator for testing
    rng = np.random.default_rng(seed)

    # Add the wires as nodes to the graph
    for i in range(NWN.graph["wire_num"]):
        NWN.graph["lines"].append(create_line(NWN.graph["wire_length"], xmax=NWN.graph["width"], ymax=NWN.graph["width"], rng=rng))
        NWN.add_node((i,), electrode=False)
        
    # Find intersects
    intersect_dict = find_intersects(NWN.graph["lines"])
    NWN.add_edges_from(
        [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
        resistance = resistance,
        capacitance = capacitance,
        is_open = False
    )
    NWN.graph["loc"] = intersect_dict
    
    # Find junction density
    NWN.graph["junction_density"] = len(intersect_dict) / size

    return NWN


def convert_NWN_to_MNR(NWN: nx.Graph):
    """
    Converts a NWN from the junction-dominated assumption to the 
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
        if i in NWN.graph["electrode_list"]:
            continue

        # Get junction attributes
        junction_resistance = NWN.graph["junction_resistance"]
        junction_capacitance = NWN.graph["junction_capacitance"]

        # Split nodes into subnodes representing the junctions on the wires
        for j, (edge, loc) in enumerate(junction_locs.items()):
            other_node = edge[~edge.index((i,))]
            NWN.add_node((i, j), loc=loc, electrode=False)
            NWN.add_edge(
                (i, j), other_node, resistance=junction_resistance, capacitance=junction_capacitance, is_open=False
            )
        NWN.remove_node((i,))

        # Add edges between subnodes
        for ind, next_ind in zip(ordering, ordering[1:]):
            NWN.add_edge((i, ind), (i, next_ind))


def plot_NWN(NWN, intersections=True, rnd_color=False):
    """
    Plots a given nanowire network.

    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot intersection plots if required
    if intersections:
        ax.scatter(
            *np.array([(point.x, point.y) for point in NWN.graph["loc"].values()]).T, 
            zorder=10, s=5, c="blue"
        )

    # Defaults to blue and pink lines, else random colors are used.
    if rnd_color:
        for i in range(NWN.graph["wire_num"]):
            ax.plot(*np.array(NWN.graph["lines"][i]).T)
    else:
        for i in range(NWN.graph["wire_num"]):
            if i in NWN.graph["electrode_list"]:
                ax.plot(*np.array(NWN.graph["lines"][i]).T, c="xkcd:light blue")
            else:
                ax.plot(*np.array(NWN.graph["lines"][i]).T, c="pink")

    plt.show()
    return fig, ax


def draw_NWN(
    NWN: nx.Graph, 
    figsize: tuple = None,
    font_size: int = 8,
    sol: np.ndarray = None
):
    """
    Draw the given nanowire network as a networkx graph.

    """
    fig, ax = plt.subplots(figsize=figsize)

    if NWN.graph["type"] == "JDA":
        # Nodes are placed at the center of the wire
        pos = {(i,): np.array(NWN.graph["lines"][i].centroid) for i in range(NWN.graph["wire_num"])}

        # Label node voltages if sol is given, else just label as nodes numbers
        if sol is not None:
            labels = {(key,): str(round(value, 2)) for key, value in zip(range(NWN.graph["wire_num"]), sol)}
        else:
            labels = {(i,): i for i in range(NWN.graph["wire_num"])}

        nx.draw(NWN, ax=ax, node_size=40, pos=pos, labels=labels, font_size=font_size, edge_color="r")

    elif NWN.graph["type"] == "MNR":
        pass

    else:
        raise ValueError("Nanowire network has invalid type.")

    plt.show()
    return fig, ax


def add_wires(NWN: nx.Graph, lines: List[LineString], electrodes: List[bool]):
    """
    Adds wires to a given nanowire network.

    Currently, adding a wire that already exists breaks things.

    Also, adding wires to a MNR NWN does not work yet.
    
    """
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
        
        if electrodes[i]:
            NWN.graph["electrode_list"].append(start_ind + i)

        # Find intersects
        intersect_dict = find_line_intersects(start_ind + i, NWN.graph["lines"])

        # Custom contact junction resistances
        # for ind in intersect_dict.keys():
        #     resistance = NWN.graph["junction_resistance"]
        #     if ind[0] in NWN.graph["electrode_list"] or ind[1] in NWN.graph["electrode_list"]:
        #         resistance = 0.0
        #     NWN.add_edge(*ind, resistance=resistance)
        
        # Uniform junction resistances
        NWN.add_edges_from(
            [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
            resistance = NWN.graph["junction_resistance"],
            capacitance = NWN.graph["junction_capacitance"],
            is_open = False
        )
        NWN.graph["loc"].update(intersect_dict)

    # Update wire density
    NWN.graph["wire_density"] = (NWN.graph["wire_num"] - len(NWN.graph["electrode_list"])) / NWN.graph["size"]
