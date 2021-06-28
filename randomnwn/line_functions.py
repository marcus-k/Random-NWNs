#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Functions for random lines.
# 
# Author: Marcus Kasdorf
# Date:   May 20, 2021

from typing import List, Dict, Tuple
import numpy as np
from numpy.random import uniform
from shapely.geometry import LineString, Point


def create_line(length=1, xmin=0, xmax=1, ymin=0, ymax=1, rng=None) -> LineString:
    """
    Generate random lines with random orientations with midpoints ranging from 
    area from ``xmin`` to ``xmax`` and from ``ymin`` to ``ymax``.

    Parameters
    ----------
    length : float
        Length of line
    
    xmin : float
        Minimum x coordinate midpoint.

    xmax : float
        Minimum x coordinate midpoint.

    ymin : float
        Minimum y coordinate midpoint.

    ymax : float
        Minimum y coordinate midpoint.

    rng : Generator
        Generator object usually created from ``default_rng`` from 
        ``numpy.random``. A seeded generator can be passed for consistent 
        random numbers. If None, uses the default NumPy random functions.

    Returns
    -------
    out : LineString
        LineString of the generated line.

    """
    if rng is not None:
        xmid, ymid, angle = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax), rng.uniform(0, np.pi)
    else:
        xmid, ymid, angle = uniform(xmin, xmax), uniform(ymin, ymax), uniform(0, np.pi)

    xhalf, yhalf = length / 2 * np.cos(angle), length / 2 * np.sin(angle)

    xstart, xend = xmid - xhalf, xmid + xhalf
    ystart, yend = ymid - yhalf, ymid + yhalf

    out = LineString([(xstart, ystart), (xend, yend)])
    return out


def find_intersects(lines: list) -> Dict[Tuple[int, int], Point]:
    """
    Given a list of LineStrings, finds all the lines that intersect and where.

    Parameters
    ----------
    lines : list of LineStrings
        List of the LineStrings to find the intersections of.

    loc : bool
        Whether or not to return the intersect locations. Defaults to false.

    Returns
    -------
    out : dict
        Dictionary where the key is a tuple of the pair of intersecting lines 
        and the value is the intersection locations.

    """
    out = {}

    for i, j in zip(*np.triu_indices(n=len(lines), k=1)):
        # Check for intersection first before calculating it
        if lines[i].intersects(lines[j]):
            out.update({(i, j): lines[i].intersection(lines[j])})

    return out


def find_line_intersects(ind: int, lines: List[LineString]) -> Dict[Tuple[int, int], Point]:
    """
    Given a list of LineStrings, find all the lines that intersect
    with a specified line in the list given by the index ``ind``.

    """
    out = {}

    for j in range(len(lines)):
        # Skip intersection with the line itself
        if ind == j:
            continue
        
        # Checking if these's an intersection first is faster
        if lines[ind].intersects(lines[j]):
            if ind < j:
                out.update({(ind, j): lines[ind].intersection(lines[j])})
            else:
                out.update({(j, ind): lines[ind].intersection(lines[j])})

    return out


def add_points_to_line(line: LineString, points: List[Point], return_ordering=False):
    """
    Given a list of points and a line, add the projected points to the line.

    See general form at: 
    https://stackoverflow.com/questions/34754777/shapely-split-linestrings-at-intersections-with-other-linestrings

    """
     # First coords of line (start + end)
    coords = [line.coords[0], line.coords[-1]]

    # Add the coords from the points
    coords += [(p.x, p.y) for p in points]

    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]

    # Sort the coords based on the distances
    coords, ordering = map(list, zip(
        *[(point, ind) for _, point, ind in sorted(zip(dists, coords, range(len(coords))))]
    ))

    ordering.remove(0); ordering.remove(1)
    ordering = [ordering[i] - 2 for i in range(len(ordering))]

    # Overwrite old line
    line = LineString(coords)

    if return_ordering:
        return line, ordering
    else:
        return line
    

