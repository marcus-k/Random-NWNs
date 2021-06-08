#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions to create nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   June 4, 2021

from typing import List, Union
from numbers import Number
import numpy as np
from shapely.geometry import LineString
import networkx as nx
import matplotlib.pyplot as plt

from .line_functions import *


def create_NWN(
    wire_length: float = 7.0,
    size: Union[tuple, float] = 50.0,
    density: float = 0.3, 
    seed: int = None,
    conductance: float = 0.1,
    capacitance: float = 1000,
    diameter: float = 50.0,
    resistivity: float = 22.6,
    break_voltage: float = -1
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
        both dimensions will have the same length.

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
        type = "JDA"
    )

    # Create seeded random generator for testing
    rng = np.random.default_rng(seed)

    # Add the wires as nodes to the graph
    for i in range(NWN.graph["wire_num"]):
        NWN.graph["lines"].append(create_line(
            NWN.graph["wire_length"], xmax=NWN.graph["length"], ymax=NWN.graph["width"], rng=rng
        ))
        NWN.add_node((i,), electrode=False)
        
    # Find intersects
    intersect_dict = find_intersects(NWN.graph["lines"])
    NWN.add_edges_from(
        [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
        conductance = conductance,
        is_shorted = conductance,
        capacitance = capacitance,
        type = "junction"
    )
    NWN.graph["loc"] = intersect_dict
    
    # Find junction density
    NWN.graph["junction_density"] = len(intersect_dict) / size

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
        if i in NWN.graph["electrode_list"]:
            continue

        # Split nodes into subnodes representing the junctions on the wires
        for j, (edge, loc) in enumerate(junction_locs.items()):
            # Find connecting node
            other_node = edge[~edge.index((i,))]

            # Get old edge attributes
            old_attributes = NWN.edges[edge]

            # Create replacing MNR node and edge
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
                is_shorted = wire_conductance,
                capacitance = 0, 
                type = "inner"
            )


def plot_NWN(NWN, intersections=True, rnd_color=False):
    """
    Plots a given nanowire network.

    Parameters
    ----------
    NWN : Graph
        Nanowire network to plot.

    intersections : bool, optional
        Whether or not to scatter plot the interesections as well.
        Defaults to true.

    rnd_color : bool, optional
        Whether or not to randomize the colors of the plotted lines.
        Defaults to false.

    Returns
    -------
    fig : Figure
        Figure object of the plot.

    ax : Axes
        Axes object of the plot.

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

    Parameters
    ----------
    NNW : Graph
        Nanowire network to draw.

    figsize : tuple, optional
        Figure size to be passed to `plt.subplots`.

    font_size : int, optional
        Font size to be passed to `nx.draw`.

    sol : ndarray, optional
        If supplied, these values will be display as node labels
        instead of the names of the nodes.

    Returns
    -------
    fig : Figure
        Figure object of the plot.

    ax : Axes
        Axes object of the plot.

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
        kwargs = {}
        if sol is not None:
            labels = {node: str(round(value, 2)) for node, value in zip(sorted(NWN.nodes()), sol)}
            kwargs.update({"labels": labels})
        else:
            kwargs.update({"with_labels": True})


        nx.draw(NWN, ax=ax, node_size=40, font_size=font_size, edge_color="r", **kwargs)

    else:
        raise ValueError("Nanowire network has invalid type.")

    plt.show()
    return fig, ax


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
        
        if electrodes[i]:
            NWN.graph["electrode_list"].append(start_ind + i)

        # Find intersects
        intersect_dict = find_line_intersects(start_ind + i, NWN.graph["lines"])
        
        # Add edges to NWN
        conductance = 1 / resistance if resistance is not None else NWN.graph["junction_conductance"]
        NWN.add_edges_from(
            [((key[0],), (key[1],)) for key in intersect_dict.keys()], 
            conductance = conductance,
            is_shorted = conductance,
            capacitance = NWN.graph["junction_capacitance"],
            type = "junction"
        )
        NWN.graph["loc"].update(intersect_dict)

    # Update wire density
    NWN.graph["wire_density"] = (NWN.graph["wire_num"] - len(NWN.graph["electrode_list"])) / NWN.graph["size"]


def set_junction_resistance(NWN: nx.Graph, R: float):
    """
    Sets the junction resistance for a given nanowire network.

    """
    NWN.graph["junction_conductance"] = 1 / R
    attrs = {
        edge: {
            "conductance": 1 / R, "is_shorted": 1 / R
        } for edge in NWN.edges() if NWN.edges[edge]["type"] == "junction"
    }
    nx.set_edge_attributes(NWN, attrs)