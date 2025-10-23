# Random NWNs [![Tests](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml/badge.svg)](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml)

Python package for modelling and analyzing random nanowire networks. This package was a summer research project lasting from May 2021 to August 2021 under the supervision of Dr. Claudia Gomes da Rocha. 

This repository is discontinued, to be replaced its successor with: [https://github.com/marcus-k/MemNNetSim](https://github.com/marcus-k/MemNNetSim)

# Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Uninstallation](#uninstallation)

# Installation

Random NWNs can be installed from PyPI for quick use or installed manually for development.

## Production

The latest version of randomnwn can be installed from PyPI:

`pip install randomnwn`

An Anaconda environment file is also provided to create a new virtual 
environment with the minimum required dependencies required to run the package.

`conda env create -n randomnwn -f environment.yml`

Be sure you activate the environment before using the package!

`conda activate randomnwn`

## Development

One can use the `dev-environment.yml` file with Anaconda to create a new 
virtual environment with all the required dependencies for development.

`conda env create -n randomnwn -f dev-environment.yml`

This will also install the randomnwn package in editable mode (i.e. as if 
running `pip install -e .` in the base folder).

# Usage

Nanowire network objects are simply [NetworkX](https://github.com/networkx/networkx) graphs with various attributes stored in the graph, edges, and nodes.

```python
>>> import randomnwn as rnwn
>>> NWN = rnwn.create_NWN(seed=123)
>>> NWN
                Type: JDA
               Wires: 750
          Electrodes: 0
Inner-wire junctions: None
      Wire junctions: 3238
              Length: 50.00 um (7.143 l0)
               Width: 50.00 um (7.143 l0)
        Wire Density: 0.3000 um^-2 (14.70 l0^-2)
>>> rnwn.plot_NWN(NWN)
(<Figure size 800x600 with 1 Axes>, <AxesSubplot:>)
```
![Figure_1](https://user-images.githubusercontent.com/81660172/127204015-9f882ef5-dca3-455d-998f-424a5787b141.png)

See the [wiki pages](https://github.com/Marcus-Repository/Random-NWNs/wiki) for more detail on usage.

# Uninstallation

To uninstall the package, use:

`pip uninstall randomnwn`
