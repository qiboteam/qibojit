from qibo.config import raise_error

from qibojit.backends.cpu import NumbaBackend
from qibojit.backends.gpu import CupyBackend, CuQuantumBackend

QibojitBackend = NumbaBackend | CupyBackend | CuQuantumBackend


class MetaBackend:
    """Meta-backend class which takes care of loading the qibojit backends."""

    PLATFORMS = ("numba", "cupy", "cuquantum")

    @staticmethod
    def load(platform: str) -> QibojitBackend:
        """Loads the backend.

        Args:
            platform (str): Name of the backend to load: either `numba`, `cupy` or `cuquantum`.
        Returns:
            qibo.backends.abstract.AbstractBackend: The loaded backend.
        """

        if platform == "numba":
            return NumbaBackend()
        elif platform == "cupy":
            return CupyBackend()
        elif platform == "cuquantum":
            return CuQuantumBackend()
        else:
            raise_error(
                ValueError,
                f"Unsupported platform, please use one among {PLATFORMS}.",
            )

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
