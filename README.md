# Random NWNs [![Tests](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml/badge.svg)](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml)

Python package for modelling and analyzing random nanowire networks. This package was a summer research project going from May 2021 to August 2021.

# Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Uninstallation](#uninstallation)

# Installation

Random NWNs can be installed from PyPI for quick use or installed manually for development.

## Latest

The latest version can be installed from PyPI:

`pip install randomnwn`

## Development

Download this repository, then navigate to the base folder and run:

`pip install .`

To install the package in editable mode instead (i.e. using the local project
path), one can use:

`pip install -e .`

# Usage

Nanowire network objects are simply [NetworkX](https://github.com/networkx/networkx) graphs with various attributes stored in the graph, edges, and nodes.

```python
>>> import randomnwn as rnwn
>>> NWN = rnwn.create_NWN(seed=123)
>>> NWN
<networkx.classes.graph.Graph at 0x...>
>>> rnwn.plot_NWN(NWN)
(<Figure size 800x600 with 1 Axes>, <AxesSubplot:>)
```
![Figure_1](https://user-images.githubusercontent.com/81660172/127204015-9f882ef5-dca3-455d-998f-424a5787b141.png)

See the [wiki pages](https://github.com/Marcus-Repository/Random-NWNs/wiki) for more detail on usage.

# Uninstallation

To uninstall the package, use:

`pip uninstall randomnwn`
