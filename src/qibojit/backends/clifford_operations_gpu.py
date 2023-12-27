import cupy as cp
import numpy as np
import qibo.backends.clifford_operations as co
from cupyx import jit
from qibo.backends.clifford_operations import *


@jit.rawkernel()
def H(symplectic_matrix, q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        tmp = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = tmp
    return symplectic_matrix


@jit.rawkernel()
def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, control_q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, control_q] & z[i, target_q]) & (
            x[i, target_q] ^ ~z[i, control_q]
        )
        symplectic_matrix[i, target_q] = x[i, target_q] ^ x[i, control_q]
        symplectic_matrix[i, nqubits + control_q] = z[i, control_q] ^ z[i, target_q]
    return symplectic_matrix


@jit.rawkernel()
def CZ(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, control_q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
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


@jit.rawkernel()
def S(symplectic_matrix, q, nqubits):
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & z[i, q])
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@jit.rawkernel()
def Z(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (
            (x[i, q] & z[i, q]) ^ x[i, q] & (z[i, q] ^ x[i, q])
        )
    return symplectic_matrix


@jit.rawkernel()
def X(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (z[i, q] & x[i, q])
        )
    return symplectic_matrix


@jit.rawkernel()
def Y(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-H-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = (
            r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q])) ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        )
    return symplectic_matrix


@jit.rawkernel()
def SX(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@jit.rawkernel()
def SDG(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        symplectic_matrix[i, nqubits + q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@jit.rawkernel()
def SXDG(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-S-H"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & x[i, q])
        symplectic_matrix[i, q] = z[i, q] ^ x[i, q]
    return symplectic_matrix


@jit.rawkernel()
def RY_pi(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (x[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@jit.rawkernel()
def RY_3pi_2(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
        symplectic_matrix[i, -1] = r[i] ^ (z[i, q] & (z[i, q] ^ x[i, q]))
        zq = symplectic_matrix[i, nqubits + q]
        symplectic_matrix[i, nqubits + q] = symplectic_matrix[i, q]
        symplectic_matrix[i, q] = zq
    return symplectic_matrix


@jit.rawkernel()
def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> CNOT-CNOT-CNOT"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, control_q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
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


@jit.rawkernel()
def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, control_q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
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


@jit.rawkernel()
def CY(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> S-CNOT-SDG"""
    r = symplectic_matrix[:-1, -1]
    x = symplectic_matrix[:-1, :nqubits]
    z = symplectic_matrix[:-1, nqubits:-1]
    xq = x[:, control_q]
    tid = jit.blockIdx.xq * jit.blockDim.xq + jit.threadIdx.xq
    ntid = jit.gridDim.xq * jit.blockDim.xq
    for i in range(tid, xq.shape[0], ntid):
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


@jit.rawkernel()
def _rowsum(symplectic_matrix, h, i, nqubits, include_scratch: bool = False):
    x = symplectic_matrix[: -1 + (2 * nqubits + 2) * int(include_scratch), :nqubits]
    z = symplectic_matrix[: -1 + (2 * nqubits + 2) * int(include_scratch), nqubits:-1]

    x1, x2 = x[i, :], x[h, :]
    z1, z2 = z[i, :], z[h, :]
    tid = jit.blockIdx.h * jit.blockDim.h + jit.threadIdx.h
    ntid = jit.gridDim.h * jit.blockDim.h
    for j in range(tid, len(h), ntid):
        exp = cp.zeros(nqubits, dtype=int64)
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
            + cp.sum(exp)
        ) % 4 != 0
        symplectic_matrix[h[j], :nqubits] = x[i[j], :] ^ x[h[j], :]
        symplectic_matrix[h[j], nqubits:-1] = z[i[j], :] ^ z[h[j], :]
    return symplectic_matrix


@jit.rawkernel()
def _determined_outcome(state, q, nqubits):
    state[-1, :] = False
    indices = state[:nqubits, q].nonzero()[0]
    tid = jit.blockIdx.indices * jit.blockDim.indices + jit.threadIdx.indices
    ntid = jit.gridDim.indices * jit.blockDim.indices
    for i in range(tid, len(indices), ntid):
        state = _rowsum(
            state,
            np.array([2 * nqubits], dtype=uint64),
            np.array([indices[i] + nqubits], dtype=uint64),
            nqubits,
            include_scratch=True,
        )
    return state, uint64(state[-1, -1])


# monkey-patching the original qibo clifford operations
for f in [
    "H",
    "CNOT",
    "CZ",
    "S",
    "Z",
    "X",
    "Y",
    "SX",
    "SDG",
    "SXDG",
    "RY_pi",
    "RY_3pi_2",
    "SWAP",
    "iSWAP",
    "CY",
    "_rowsum",
    "_determined_outcome",
]:
    setattr(co, f, locals()[f])
