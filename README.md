# Random NWNs [![Tests](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml/badge.svg)](https://github.com/marcus-k/Random-NWNs/actions/workflows/python-package.yml)

Python package for modelling and analyzing random nanowire networks. This package was a summer research project lasting from May 2021 to August 2021 under the supervision of Dr. Claudia Gomes da Rocha. 

**Update:** This project will now be continuing as of May 2024. If you are using this project, please note there will be **active development** on it and the functionality may change.

For future additions, feel free to fork the repository. Please cite Marcus Kasdorf if you wish to extend the project.

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

For convenience, one can use the `environment.yml` file with Anaconda to create a new
virtual environment with all the required dependencies.

`conda env create -f environment.yml`

This will create a new environment named `randomnwn`. To activate the environment, use:

`conda activate randomnwn`

Then, to install the package, use pip. One can install the package in the usual way
above, or install it in editable mode to allow for local development. Navigate to the 
base folder and run:

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
