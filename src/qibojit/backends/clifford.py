from numba import njit
from qibo.backends.clifford import CliffordOperations as CO


class CliffordOperations(CO):
    def __init__(self, engine):
        super().__init__(engine)

    @staticmethod
    def I(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.I)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def H(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.H)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def CNOT(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(CO.CNOT)(
            symplectic_matrix, control_q, target_q, nqubits
        )

    @staticmethod
    def CZ(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(CO.CZ)(
            symplectic_matrix, control_q, target_q, nqubits
        )

    @staticmethod
    def S(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.S)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def Z(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.Z)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def X(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.X)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def Y(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.Y)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def SX(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.SX)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def SDG(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.SDG)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def SXDG(symplectic_matrix, q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)(CO.SXDG)(
            symplectic_matrix, q, nqubits
        )

    @staticmethod
    def RX(symplectic_matrix, q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, f4)", parallel=True, cache=True)(CO.RX)(
            symplectic_matrix, q, nqubits, theta
        )

    @staticmethod
    def RZ(symplectic_matrix, q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, f4)", parallel=True, cache=True)(CO.RZ)(
            symplectic_matrix, q, nqubits, theta
        )

    @staticmethod
    def RY(symplectic_matrix, q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, f4)", parallel=True, cache=True)(CO.RY)(
            symplectic_matrix, q, nqubits, theta
        )

    @staticmethod
    def SWAP(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(CO.SWAP)(
            symplectic_matrix, control_q, target_q, nqubits
        )

    @staticmethod
    def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(
            CO.iSWAP
        )(symplectic_matrix, control_q, target_q, nqubits)

    @staticmethod
    def FSWAP(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(
            CO.FSWAP
        )(symplectic_matrix, control_q, target_q, nqubits)

    @staticmethod
    def CY(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(CO.CY)(
            symplectic_matrix, control_q, target_q, nqubits
        )

    @staticmethod
    def CRX(symplectic_matrix, control_q, target_q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, u8, f4)", parallel=True, cache=True)(
            CO.CRX
        )(symplectic_matrix, control_q, target_q, nqubits, theta)

    @staticmethod
    def CRZ(symplectic_matrix, control_q, target_q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, u8, f4)", parallel=True, cache=True)(
            CO.CRZ
        )(symplectic_matrix, control_q, target_q, nqubits, theta)

    @staticmethod
    def CRY(symplectic_matrix, control_q, target_q, nqubits, theta):
        return njit("b1[:,:](b1[:,:], u8, u8, u8, f4)", parallel=True, cache=True)(
            CO.CRY
        )(symplectic_matrix, control_q, target_q, nqubits, theta)

    @staticmethod
    def ECR(symplectic_matrix, control_q, target_q, nqubits):
        return njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)(CO.ECR)(
            symplectic_matrix, control_q, target_q, nqubits
        )
