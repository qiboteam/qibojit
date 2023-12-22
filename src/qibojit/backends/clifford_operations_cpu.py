import numpy as np
from numba import njit, prange, uint64
from qibo.backends.clifford_operations import M as _M
from qibo.backends.clifford_operations import *


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def H(symplectic_matrix, q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        tmp = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = tmp
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)
def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, control_q] & z[i, target_q]) & (
            x[i, target_q] ^ ~z[i, control_q]
        )
        symplectic_matrix[i, target_q] = x[i, target_q] ^ x[i, control_q]
        symplectic_matrix[i, nqubits + control_q] = z[i, control_q] ^ z[i, target_q]
    return symplectic_matrix


@staticmethod
@njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)
def CZ(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = (
            r[i]
            ^ (x[i, target_q] & z[i, target_q])
            ^ (x[i, control_q] & x[i, target_q] & (z[i, target_q] ^ ~z[i, control_q]))
            ^ (x[i, target_q] & (z[i, target_q] ^ x[i, control_q]))
        )
        z_control_q = x[i, target_q] ^ z[i, control_q]
        z_target_q = z[i, target_q] ^ x[i, control_q]
        symplectic_matrix[i, nqubits + control_q] = z_control_q
        symplectic_matrix[i, nqubits + target_q] = z_target_q
    return symplectic_matrix


@staticmethod
@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def S(symplectic_matrix, q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def Z(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (
            (x[i, q] & z[i, q]) ^ x[i, q] & (z[i, q] ^ x[i, q])
        )
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def X(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (z[i, q] & x[i, q])
        )
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def Y(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-H-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        )
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def SX(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def SDG(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def SXDG(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & x[i, q])
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def RY_pi(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8)", parallel=True, cache=True)
def RY_3pi_2(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)
def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> CNOT-CNOT-CNOT"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[:-1, -1] = (
            r[i]
            ^ (x[i, control_q] & z[i, target_q] & (x[i, target_q] ^ ~z[i, control_q]))
            ^ (
                (x[i, target_q] ^ x[i, control_q])
                & (z[i, target_q] ^ z[i, control_q])
                & (z[i, target_q] ^ ~x[i, control_q])
            )
            ^ (
                x[i, target_q]
                & z[i, control_q]
                & (x[i, control_q] ^ x[i, target_q] ^ z[i, control_q] ^ ~z[i, target_q])
            )
        )
        x_cq = symplectic_matrix[i, control_q]
        x_tq = symplectic_matrix[i, target_q]
        z_cq = symplectic_matrix[i, nqubits + control_q]
        z_tq = symplectic_matrix[i, nqubits + target_q]
        symplectic_matrix[i, control_q] = x_tq
        symplectic_matrix[i, target_q] = x_cq
        symplectic_matrix[i, nqubits + control_q] = z_tq
        symplectic_matrix[i, nqubits + target_q] = z_cq
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)
def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = (
            r[i]
            ^ (x[i, target_q] & z[i, target_q])
            ^ (x[i, control_q] & z[i, control_q])
            ^ (x[i, control_q] & (z[i, control_q] ^ x[i, control_q]))
            ^ (
                (z[i, control_q] ^ x[i, control_q])
                & (z[i, target_q] ^ x[i, target_q])
                & (x[i, target_q] ^ ~x[i, control_q])
            )
            ^ (
                (x[i, target_q] ^ z[i, control_q] ^ x[i, control_q])
                & (x[i, target_q] ^ z[i, target_q] ^ x[i, control_q])
                & (x[i, target_q] ^ z[i, target_q] ^ x[i, control_q] ^ ~z[i, control_q])
            )
            ^ (x[i, control_q] & (x[i, target_q] ^ x[i, control_q] ^ z[i, control_q]))
        )
        z_control_q = x[i, target_q] ^ z[i, target_q] ^ x[i, control_q]
        z_target_q = x[i, target_q] ^ z[i, control_q] ^ x[i, control_q]
        symplectic_matrix[i, nqubits + control_q] = z_control_q
        symplectic_matrix[i, nqubits + target_q] = z_target_q
        tmp = symplectic_matrix[i, control_q]
        symplectic_matrix[i, control_q] = symplectic_matrix[i, target_q]
        symplectic_matrix[i, target_q] = tmp
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8, u8, u8)", parallel=True, cache=True)
def CY(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> S-CNOT-SDG"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    for i in prange(symplectic_matrix.shape[0] - 1):
        symplectic_matrix[i, -1] = (
            r[i]
            ^ (x[i, target_q] & (z[i, target_q] ^ x[i, target_q]))
            ^ (
                x[i, control_q]
                & (x[i, target_q] ^ z[i, target_q])
                & (z[i, control_q] ^ ~x[i, target_q])
            )
            ^ ((x[i, target_q] ^ x[i, control_q]) & (z[i, target_q] ^ x[i, target_q]))
        )
        x_target_q = x[i, control_q] ^ x[i, target_q]
        z_control_q = z[i, control_q] ^ z[i, target_q] ^ x[i, target_q]
        z_target_q = z[i, target_q] ^ x[i, control_q]
        symplectic_matrix[i, target_q] = x_target_q
        symplectic_matrix[i, nqubits + control_q] = z_control_q
        symplectic_matrix[i, nqubits + target_q] = z_target_q
    return symplectic_matrix


@njit("b1[:,:](b1[:,:], u8[:], u8[:], u8, b1)", parallel=True, cache=True)
def _rowsum(symplectic_matrix, h, i, nqubits, include_scratch: bool = False):
    print("using numba rowsum")
    x = symplectic_matrix[: -1 + (2 * nqubits + 2) * uint64(include_scratch), :nqubits]
    z = symplectic_matrix[
        : -1 + (2 * nqubits + 2) * uint64(include_scratch), nqubits:-1
    ]

    x1, x2 = x[i, :], x[h, :]
    z1, z2 = z[i, :], z[h, :]
    for j in prange(len(h)):
        exp = np.zeros(nqubits, dtype=uint64)
        x1_eq_z1 = (x1[j] ^ z1[j]) == False
        x1_neq_z1 = ~x1_eq_z1
        x1_eq_0 = x1[j] == False
        x1_eq_1 = ~x1_eq_0
        ind2 = x1_eq_z1 & x1_eq_1
        ind3 = x1_eq_1 & x1_neq_z1
        ind4 = x1_eq_0 & x1_neq_z1
        exp[ind2] = z2[j, ind2].astype(uint64) - x2[j, ind2].astype(uint64)
        exp[ind3] = z2[j, ind3].astype(uint64) * (2 * x2[j, ind3].astype(uint64) - 1)
        exp[ind4] = x2[j, ind4].astype(uint64) * (1 - 2 * z2[j, ind4].astype(uint64))

        symplectic_matrix[h[j], -1] = (
            2 * symplectic_matrix[h[j], -1]
            + 2 * symplectic_matrix[i[j], -1]
            + np.sum(exp)
        ) % 4 != 0
        symplectic_matrix[h[j], :nqubits] = x[i[j], :] ^ x[h[j], :]
        symplectic_matrix[h[j], nqubits:-1] = z[i[j], :] ^ z[h[j], :]
    return symplectic_matrix


@njit("Tuple((b1[:,:], u8))(b1[:,:], u8, u8)", parallel=False, cache=True)
def _determined_outcome(state, q, nqubits):
    state[-1, :] = False
    indices = state[:nqubits, q].nonzero()[0]
    for i in prange(len(indices)):
        state = _rowsum(
            state,
            np.array([2 * nqubits], dtype=uint64),
            np.array([indices[i] + nqubits], dtype=uint64),
            nqubits,
            include_scratch=True,
        )
    return state, uint64(state[-1, -1])
