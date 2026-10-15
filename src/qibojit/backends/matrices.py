from functools import cached_property  # pylint: disable=E0611

import numpy as np
from qibo.backends.npmatrices import NumpyMatrices


class CustomMatrices(NumpyMatrices):
    """Matrices used by custom operators.

    They may not correspond to the mathematical representation of each gate.
    """

    def CRX(self, theta):  # noqa: N802
        return self.RX(theta)

    def CRY(self, theta):  # noqa: N802
        return self.RY(theta)

    def CRZ(self, theta):  # noqa: N802
        return self.RZ(theta)

    def CU2(self, phi, lam):  # noqa: N802
        return self.U2(phi, lam)

    def CU3(self, theta, phi, lam):  # noqa: N802
        return self.U3(theta, phi, lam)

    def U1(self, theta):  # noqa: N802
        dtype = getattr(np, self.dtype)
        return self._cast(np.exp(1j * theta), dtype=dtype)

    def CU1(self, theta):  # noqa: N802
        return self.U1(theta)

    @cached_property
    def CZ(self):  # noqa: N802
        return self.Z

    @cached_property
    def CCZ(self):  # noqa: N802
        return self.Z

    @cached_property
    def CY(self):  # noqa: N802
        return self.Y

    @cached_property
    def CH(self):  # noqa: N802
        return self.H

    @cached_property
    def CSX(self):  # noqa: N802
        return self.SX

    @cached_property
    def CSXDG(self):  # noqa: N802
        return self.SXDG

    def DEUTSCH(self, theta):  # noqa: N802
        return 1j * self.RX(2 * theta)

    def fSim(self, theta, phi):  # noqa: N802
        cost = np.cos(theta) + 0j
        isint = -1j * np.sin(theta)
        phase = np.exp(-1j * phi)
        return np.array([cost, isint, isint, cost, phase], dtype=self.dtype)

    def GeneralizedfSim(self, u, phi):  # noqa: N802
        phase = np.exp(-1j * phi)
        return np.array([u[0, 0], u[0, 1], u[1, 0], u[1, 1], phase], dtype=self.dtype)


class CupyMatrices(NumpyMatrices):  # pragma: no cover
    """Casting NumpyMatrices to Cupy arrays."""

    def __init__(self, dtype):
        super().__init__(dtype)
        import cupy as cp  # pylint: disable=E0401

        self.cp = cp

    def I(self, n=2):  # noqa: N802, E743
        return self.cp.eye(n, dtype=self.dtype)

    def _cast(self, x, dtype):
        if not isinstance(x, list) and len(x.shape) == 0:
            return self.cp.array(x, dtype=dtype)

        is_cupy = [
            isinstance(item, self.cp.ndarray) for sublist in x for item in sublist
        ]
        if any(is_cupy) and not all(is_cupy):
            # for parametrized gates x is a mixed list of cp.arrays and floats
            # thus a simple cp.array(x) fails
            # first convert the cp.arrays to numpy, then build the numpy array and move it
            # back to GPU
            dim = len(x)
            return self.cp.array(
                np.array(
                    [
                        item.get() if isinstance(item, self.cp.ndarray) else item
                        for sublist in x
                        for item in sublist
                    ]
                ).reshape(dim, dim),
                dtype=dtype,
            )
        return self.cp.array(x, dtype=dtype)

    # Necessary to avoid https://github.com/qiboteam/qibo/issues/928
    def Unitary(self, u):  # noqa: N802
        dtype = getattr(np, self.dtype)
        return self._cast(u, dtype=dtype)


class CustomCuQuantumMatrices(CustomMatrices):  # pragma: no cover
    """Matrices used by CuQuantum custom operators."""

    @cached_property
    def CNOT(self):  # noqa: N802
        return self.X

    @cached_property
    def TOFFOLI(self):  # noqa: N802
        return self.X

    @cached_property
    def CY(self):  # noqa: N802
        return self.Y

    @cached_property
    def CZ(self):  # noqa: N802
        return self.Z

    def U1(self, theta):  # noqa: N802
        return NumpyMatrices.U1(self, theta)

    def fSim(self, theta, phi):  # noqa: N802
        return NumpyMatrices.fSim(self, theta, phi)

    def GeneralizedfSim(self, u, phi):  # noqa: N802
        return NumpyMatrices.GeneralizedfSim(self, u, phi)
