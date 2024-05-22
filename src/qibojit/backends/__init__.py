from typing import Union

from qibojit.backends.cpu import NumbaBackend
from qibojit.backends.gpu import CupyBackend, CuQuantumBackend

QibojitBackend = Union[NumbaBackend, CupyBackend, CuQuantumBackend]

PLATFORMS = ("numba", "cupy", "cuquantum")


class MetaBackend:
    """Meta-backend class which takes care of loading the qibojit backends."""

    @staticmethod
    def load(platform: str = None) -> QibojitBackend:
        """Loads the backend.

        Args:
            platform (str): Name of the backend to load: either `numba`, `cupy` or `cuquantum`.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if platform == "numba":
            return NumbaBackend()
        elif platform == "cupy":
            return CupyBackend()
        elif platform == "cuquantum":
            return CuQuantumBackend()
        else:  # pragma: no cover
            try:
                return CupyBackend()
            except (ModuleNotFoundError, ImportError):
                return NumbaBackend()

    def list_available(self) -> dict:
        """Lists all the available qibojit backends."""
        available_backends = {}
        for platform in PLATFORMS:
            try:
                MetaBackend.load(platform=platform)
                available = True
            except:
                available = False
            available_backends[platform] = available
        return available_backends
