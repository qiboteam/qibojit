"""Set of custom Numba operations for the Clifford backend."""

import numpy as np
from numba import njit, prange

PARALLEL = True


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


# this cannot be cached anymore with numba unfortunately
@njit("i8(i8)", parallel=False, cache=True)
def _packed_size(n):
    """Returns the size of an array of `n` booleans after packing."""
    return int(np.ceil(n / 8))


@njit(["u1[:,:](u1[:,:], i8)", "u1[:,:](b1[:,:], i8)"], parallel=PARALLEL, cache=True)
def _packbits(array, axis):
    array = array.astype(np.uint8)
    array = np.ascontiguousarray(np.swapaxes(array, axis, -1))
    shape = array.shape
    n = shape[-1]
    dim = 1
    for s in shape[:-1]:
        dim *= s
    array = np.reshape(array, (dim, n))
    packed_len = (n + 7) // 8
    out = np.zeros((dim, packed_len), dtype=np.uint8)
    for j in prange(dim):  # pylint: disable=not-an-iterable
        for i in prange(n):  # pylint: disable=not-an-iterable
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            out[j, byte_idx] |= array[j, i] << bit_idx
    out = np.reshape(out, shape[:-1] + (packed_len,))
    return np.swapaxes(out, axis, -1)


@njit("u1[:,:](u1[:,:], i8)", parallel=PARALLEL, cache=True)
def _pack_for_measurements(state, nqubits):
    """Prepares the state for measurements by packing the rows of the X and Z sections of the symplectic matrix."""
    r, x, z = _get_rxz(state, nqubits)
    x = _packbits(x, axis=1)
    z = _packbits(z, axis=1)
    return np.hstack((x, z, r[:, None]))


@njit("u1[:](u1)", parallel=PARALLEL, cache=True)
def _unpack_byte(byte):
    bits = np.empty(8, dtype=np.uint8)
    for i in range(8):
        bits[i] = (byte >> (7 - i)) & 1
    return bits


@njit("u1[:,:](u1[:,:], i8, i8)", parallel=PARALLEL, cache=True)
def _unpackbits(array, axis, count):
    # this is gonnna be used on 2d arrays only
    # i.e. portions of the symplectic matrix
    # thus axis is either 0 or 1
    if axis == 0:
        array = np.transpose(array, (1, 0))
    in_shape = array.shape
    byte_len = in_shape[-1]

    # Output shape: replace last dim with ceil(count / 8) * 8
    out_shape = (in_shape[0], count)
    out = np.zeros(out_shape, dtype=np.uint8)

    for idx in range(in_shape[0]):
        for i in range(byte_len):
            byte = array[idx, i]
            byte_bits = _unpack_byte(byte)
            for j in range(8):
                bit_idx = i * 8 + j
                if bit_idx < count:
                    out[idx, bit_idx] = byte_bits[j]

    # Move axis back to original location
    if axis == 0:
        out = np.transpose(out, (1, 0))
    return out


@njit("u1[:,:](u1[:,:], i8)", parallel=PARALLEL, cache=True)
def _unpack_for_measurements(state, nqubits):
    """Unpacks the symplectc matrix that was packed for measurements."""
    x = _unpackbits(state[:, : _packed_size(nqubits)], axis=1, count=nqubits)
    z = _unpackbits(state[:, _packed_size(nqubits) : -1], axis=1, count=nqubits)
    return np.hstack((x, z, state[:, -1][:, None]))


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
    symplectic_matrix = _pack_for_measurements(symplectic_matrix, nqubits)
    packed_n = _packed_size(nqubits)
    packed_xi, packed_zi = (
        symplectic_matrix[i, :packed_n],
        symplectic_matrix[i, packed_n:-1],
    )
    packed_xh, packed_zh = (
        symplectic_matrix[h, :packed_n],
        symplectic_matrix[h, packed_n:-1],
    )
    if determined:
        g_r = np.zeros(h.shape[0], dtype=np.uint8)
        g_xi_xh = packed_xi.copy()
        g_zi_zh = packed_xi.copy()
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
        xi_xh = packed_xi[j] ^ packed_xh[j]
        zi_zh = packed_zi[j] ^ packed_zh[j]
        if determined:  # for some reason xor reduction fails here
            g_r[j] = r  # thus, I cannot do g_r += r here
            g_xi_xh[j] = xi_xh
            g_zi_zh[j] = zi_zh
        else:
            symplectic_matrix[h[j], -1] = r
            symplectic_matrix[h[j], :packed_n] = xi_xh
            symplectic_matrix[h[j], packed_n:-1] = zi_zh
    if determined:
        for j in prange(len(g_r)):  # pylint: disable=not-an-iterable
            symplectic_matrix[h[0], -1] ^= g_r[j]
            symplectic_matrix[h[0], :packed_n] ^= g_xi_xh[j]
            symplectic_matrix[h[0], packed_n:-1] ^= g_zi_zh[j]
    return _unpack_for_measurements(symplectic_matrix, nqubits)
