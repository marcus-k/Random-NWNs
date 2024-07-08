#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions to plot nanowire networks.
# 
# Author: Marcus Kasdorf
# Date:   July 8, 2021

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_NWN(
    NWN: nx.Graph, 
    intersections: bool = True, 
    rnd_color: bool = False,
    scaled: bool = False,
    grid: bool = True,
    xlabel: str = "",
    ylabel: str = "",
) -> tuple[Figure, Axes]:
    """
    Plots a given nanowire network and returns the figure and axes.

    Parameters
    ----------
    NWN : Graph
        Nanowire network to plot.

    intersections : bool, optional
        Whether or not to scatter plot the intersections as well.
        Defaults to true.

    rnd_color : bool, optional
        Whether or not to randomize the colors of the plotted lines.
        Defaults to false.

    scaled: bool, optional
        Whether or not to scale the plot by the characteristic values of the
        given nanowire network. Defaults to False.

    grid: bool, optional
        Grid lines on plot. Defaults to true.

    xlabel: str, optional
        x label string.

    ylabel: str, optional
        y label string.

    Returns
    -------
    fig : Figure
        Figure object of the plot.

    ax : Axes
        Axes object of the plot.

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    l0 = NWN.graph["units"]["l0"]

    # Plot intersection plots if required
    if intersections:
        ax.scatter(
            *np.array([(point.x, point.y) for point in NWN.graph["loc"].values()]).T, 
            zorder=10, s=5, c="blue"
        )

    # Defaults to blue and pink lines, else random colors are used.
    if rnd_color:
        for i in range(NWN.graph["wire_num"]):
            ax.plot(*np.asarray(NWN.graph["lines"][i].coords).T)
    else:
        for i in range(NWN.graph["wire_num"]):
            if (i,) in NWN.graph["electrode_list"]:
                ax.plot(*np.asarray(NWN.graph["lines"][i].coords).T, c="xkcd:light blue")
            else:
                ax.plot(*np.asarray(NWN.graph["lines"][i].coords).T, c="pink")

    # Scale axes according to the characteristic values
    if scaled:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x * l0:.3g}")
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, pos: f"{y * l0:.3g}")
        )

    # Other attributes
    if grid: 
        ax.grid(alpha=0.25)
    if xlabel: 
        ax.set_xlabel(xlabel)
    if ylabel: 
        ax.set_ylabel(ylabel)

    plt.show()
    return fig, ax


def draw_NWN(
    NWN: nx.Graph, 
    figsize: tuple = None,
    font_size: int = 8,
    node_labels: np.ndarray = None,
    fmt: str = ".2f",
    edge_colors: np.ndarray = None,
    cbar_label: str = "Colorbar",
    cmap = plt.cm.RdYlBu_r,
) -> tuple[Figure, Axes]:
    """
    Draw the given nanowire network as a networkx graph. JDA drawing is more
    detailed as nodes can be given spacial locations. With MNR drawing, nodes
    will have random locations.

    Parameters
    ----------
    NNW : Graph
        Nanowire network to draw.

    figsize : tuple, optional
        Figure size to be passed to `plt.subplots`.

    font_size : int, optional
        Font size to be passed to `nx.draw`.

    node_labels : ndarray, optional
        If supplied, these values will be display as node labels
        instead of the names of the nodes.

    fmt : str, optional
        String formatting for node labels. Only used if sol is passed.
        Default: ".2f".

    edge_colors : ndarray, optional
        List of values to color the edges. Edges are assumed to be in the
        same order as `NWN.edges`.

    cbar_label : str, optional
        Label for the colorbar.

    cmap : colormap, optional
        Matplotlib color map to use for the edges.

    Returns
    -------
    fig : Figure
        Figure object of the plot.

    ax : Axes
        Axes object of the plot.

    """
    fig, ax = plt.subplots(figsize=figsize)

    if NWN.graph["type"] == "JDA":
        kwargs = dict()

        # Nodes are placed at the center of the wire
        kwargs.update({
            "pos": {(i,): np.asarray(*NWN.graph["lines"][i].centroid.coords) 
                for i in range(NWN.graph["wire_num"])}
        })

        # Label node voltages if sol is given, else just label as nodes numbers
        if node_labels is not None:
            kwargs.update({
                "labels": {(key,): f"{value:{fmt}}" 
                    for key, value in zip(range(NWN.graph["wire_num"]), node_labels)}
            })
        else:
            kwargs.update({
                "labels": {(i,): i for i in range(NWN.graph["wire_num"])}
            })

        # Add edges colors if weights are passed
        if edge_colors is not None:
            kwargs.update({
                "edgelist": NWN.edges, 
                "edge_color": edge_colors, 
                "edge_cmap": cmap
            })

            # Add a colorbar to the network plot
            norm = mpl.colors.Normalize(
                vmin=np.nanmin(edge_colors), vmax=np.nanmax(edge_colors))

            cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
            cb = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)
            cb.set_label(cbar_label)

        else:
            kwargs.update({"edge_color": "r"})

        # Add node formatting
        kwargs.update({"ax": ax, "font_size": font_size, "node_size": 40})

        nx.draw(NWN, **kwargs)

    elif NWN.graph["type"] == "MNR":
        kwargs = {}
        if node_labels is not None:
            labels = {node: f"{value:{fmt}}" for node, value in zip(sorted(NWN.nodes()), node_labels)}
            kwargs.update({"labels": labels})
        else:
            kwargs.update({"with_labels": True})

        nx.draw(NWN, ax=ax, node_size=40, font_size=font_size, edge_color="r", **kwargs)

    else:
        raise ValueError("Nanowire network has invalid type.")

    plt.show()
    return fig, ax

