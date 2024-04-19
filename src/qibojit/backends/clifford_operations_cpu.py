"""Set of custom Numba operations for the Clifford backend."""

import numpy as np
from numba import njit, prange, uint64

PARALLEL = False


@njit("Tuple((u1[:], u1[:,:], u1[:,:]))(u1[:,:], u8)", parallel=PARALLEL, cache=True)
def _get_rxz(symplectic_matrix, nqubits):
    return (
        symplectic_matrix[:, -1],
        symplectic_matrix[:, :nqubits],
        symplectic_matrix[:, nqubits:-1],
    )


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def H(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        tmp = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = tmp
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8, u8)", parallel=PARALLEL, cache=True)
def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        r[i] = r[i] ^ (x[i, control_q] & z[i, target_q]) & (
            x[i, target_q] ^ ~z[i, control_q]
        )
        x[i, target_q] = x[i, target_q] ^ x[i, control_q]
        z[i, control_q] = z[i, control_q] ^ z[i, target_q]
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8, u8)", parallel=PARALLEL, cache=True)
def CZ(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = (
            r[i]
            ^ (x[i, target_q] & z[i, target_q])
            ^ (x[i, control_q] & x[i, target_q] & (z[i, target_q] ^ ~z[i, control_q]))
            ^ (x[i, target_q] & (z[i, target_q] ^ x[i, control_q]))
        )
        z_control_q = x[i, target_q] ^ z[i, control_q]
        z_target_q = z[i, target_q] ^ x[i, control_q]
        z[i, control_q] = z_control_q
        z[i, target_q] = z_target_q
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def S(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def Z(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (
            (x[i, q] & z[i, q]) ^ x[i, q] & (z[i, q] ^ x[i, q])
        )
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def X(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (z[i, q] & x[i, q])
        )
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def Y(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        )
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def SX(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def SDG(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def SXDG(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & x[i, q])
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def RY_pi(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8)", parallel=PARALLEL, cache=True)
def RY_3pi_2(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@njit("u1[:,:](u1[:,:], u8, u8, u8)", parallel=PARALLEL, cache=True)
def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> CNOT-CNOT-CNOT"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
        symplectic_matrix[i, -1] = (
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


@njit("u1[:,:](u1[:,:], u8, u8, u8)", parallel=PARALLEL, cache=True)
def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
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


@njit("u1[:,:](u1[:,:], u8, u8, u8)", parallel=PARALLEL, cache=True)
def CY(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> S-CNOT-SDG"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)

    for i in prange(symplectic_matrix.shape[0]):  # pylint: disable=not-an-iterable
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


@njit(
    [
        "u1[:,:](u1[:,:], u8[:], u8[:], u8, b1)",
        "u1[:,:](u1[:,:], u4[:], u4[:], u4, b1)",
    ],
    parallel=PARALLEL,
    cache=True,
    fastmath=True,
)
def _rowsum(symplectic_matrix, h, i, nqubits, determined=False):
    xi, zi = symplectic_matrix[i, :nqubits], symplectic_matrix[i, nqubits:-1]
    xh, zh = symplectic_matrix[h, :nqubits], symplectic_matrix[h, nqubits:-1]
    if determined:
        g_r = np.zeros(h.shape[0], dtype=np.uint8)
        g_xi_xh = xi.copy()
        g_zi_zh = xi.copy()
    for j in prange(len(h)):  # pylint: disable=not-an-iterable
        exp = (
            2 * (xi[j] * xh[j] * (zh[j] - zi[j]) + zi[j] * zh[j] * (xi[j] - xh[j]))
            - xi[j] * zh[j]
            + xh[j] * zi[j]
        )
        r = (
            2 * symplectic_matrix[h[j], -1]
            + 2 * symplectic_matrix[i[j], -1]
            + np.sum(exp)
        ) % 4 != 0
        xi_xh = xi[j] ^ xh[j]
        zi_zh = zi[j] ^ zh[j]
        if determined:  # for some reason xor reduction fails here
            g_r[j] = r  # thus, I cannot do g_r += r here
            g_xi_xh[j] = xi_xh
            g_zi_zh[j] = zi_zh
        else:
            symplectic_matrix[h[j], -1] = r
            symplectic_matrix[h[j], :nqubits] = xi_xh
            symplectic_matrix[h[j], nqubits:-1] = zi_zh
    if determined:
        for j in prange(len(g_r)):  # pylint: disable=not-an-iterable
            symplectic_matrix[h[0], -1] ^= g_r[j]
            symplectic_matrix[h[0], :nqubits] ^= g_xi_xh[j]
            symplectic_matrix[h[0], nqubits:-1] ^= g_zi_zh[j]
    return symplectic_matrix
