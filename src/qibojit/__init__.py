import importlib.metadata as im

from qibojit.backends import MetaBackend

__all__ = ["MetaBackend", "__version__"]
__version__ = im.version(__package__)
