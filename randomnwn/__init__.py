"""
Random NWNs

"""
from .version import __version__

# try:
#     from pathlib import Path
#     module_path = Path(__file__).parent.resolve()
#     speedup_file = module_path.joinpath("speedup.jl")

#     from julia import Main
#     Main.include(str(speedup_file))

#     _JULIA = True
# except ModuleNotFoundError:
#     _JULIA = False

from .nanowires import *
from .line_functions import *
from .calculations import *
from .plotting import *
from .dynamics import *
from ._units import *
