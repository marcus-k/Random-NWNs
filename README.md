# Random NWNs

Package for creating and analyzing random nanowire networks.

# Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Uninstallation](#uninstallation)

# Installation

Download this repository, then navigate to the base folder and run:

`pip install .`

To install the package in editable mode instead (i.e. using the local project
path), one can use

`pip install -e .`

The package can then be used by importing it as:

`from randomnwn import *`

# Usage

Create a nanowire network using:

`NWN = create_NWN()`

A networkx graph object is created with the following graph attributes:
- wire_length
- width
- size
- wire_density 
- wire_num
- junction_resistance 
- junction_capacitance
- electrode_list 
- lines
- type

# Uninstallation

To uninstall the package, use:

`pip uninstall Random-NWNs`