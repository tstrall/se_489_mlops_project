"""SE 489 MLOps Project.

Predict SLA violations from event sequences
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("se_489_mlops_project")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Ted Strall"
__email__ = "tstrall@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
