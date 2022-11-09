import sys

import numpy as np
from qibo.backends.npmatrices import NumpyMatrices

if sys.version_info.minor >= 8:
    from functools import cached_property  # pylint: disable=E0611
else:
    # Custom ``cached_property`` because it is not available for Python < 3.8
    from functools import lru_cache

    def cached_property(func):  # pragma: no cover
        @property
        def wrapper(self):
            return lru_cache()(func)(self)

        return wrapper


class CuQuantumMatrices(NumpyMatrices):
    # These matrices are used by the custom operators and may
    # not correspond to the mathematical representation of each gate

    @cached_property
    def CNOT(self):
        return self.X

    @cached_property
    def CZ(self):
        return self.Z

    def CRX(self, theta):
        return self.RX(theta)

    def CRY(self, theta):
        return self.RY(theta)

    def CRZ(self, theta):
        return self.RZ(theta)

    def CU1(self, theta):
        return self.U1(theta)

    def CU2(self, phi, lam):
        return self.U2(phi, lam)

    def CU3(self, theta, phi, lam):
        return self.U3(theta, phi, lam)

    @cached_property
    def TOFFOLI(self):
        return self.X


class CustomMatrices(CuQuantumMatrices):
    # These matrices are used by the custom operators and may
    # not correspond to the mathematical representation of each gate

    def U1(self, theta):
        dtype = getattr(np, self.dtype)
        return dtype(np.exp(1j * theta))

    def fSim(self, theta, phi):
        cost = np.cos(theta) + 0j
        isint = -1j * np.sin(theta)
        phase = np.exp(-1j * phi)
        return np.array([cost, isint, isint, cost, phase], dtype=self.dtype)

    def GeneralizedfSim(self, u, phi):
        phase = np.exp(-1j * phi)
        return np.array([u[0, 0], u[0, 1], u[1, 0], u[1, 1], phase], dtype=self.dtype)
