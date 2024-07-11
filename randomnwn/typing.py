#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Typing definitions for randomnwn.
# 
# Author: Marcus Kasdorf
# Date:   July 8, 2024

from typing import TypeAlias

JDANode: TypeAlias = tuple[int]
MNRNode: TypeAlias = tuple[int, int]
NWNNode: TypeAlias = JDANode | MNRNode
NWNEdge: TypeAlias = tuple[NWNNode, NWNNode]
NWNNodeIndex: TypeAlias = int
NWNEdgeIndex: TypeAlias = tuple[NWNNodeIndex, NWNNodeIndex]