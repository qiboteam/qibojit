import numpy as np
from cupyx import jit
from qibo.backends.clifford import CliffordOperations as CO


class CliffordOperations(CO):
    def __init__(self, engine):
        super().__init__(engine)

    @staticmethod
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
            symplectic_matrix[:, nqubits + q] = tmp
        return symplectic_matrix

    @staticmethod
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

    @staticmethod
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
                ^ (
                    x[i, control_q]
                    & x[i, target_q]
                    & (z[i, target_q] ^ ~z[i, control_q])
                )
                ^ (x[i, target_q] & (z[i, target_q] ^ x[i, control_q]))
            )
            z_control_q = x[i, target_q] ^ z[i, control_q]
            z_target_q = z[i, target_q] ^ x[i, control_q]
            symplectic_matrix[i, nqubits + control_q] = z_control_q
            symplectic_matrix[i, nqubits + target_q] = z_target_q
        return symplectic_matrix

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                ^ (
                    x[i, control_q]
                    & z[i, target_q]
                    & (x[i, target_q] ^ ~z[i, control_q])
                )
                ^ (
                    (x[i, target_q] ^ x[i, control_q])
                    & (z[i, target_q] ^ z[i, control_q])
                    & (z[i, target_q] ^ ~x[i, control_q])
                )
                ^ (
                    x[i, target_q]
                    & z[i, control_q]
                    & (
                        x[i, control_q]
                        ^ x[i, target_q]
                        ^ z[i, control_q]
                        ^ ~z[i, target_q]
                    )
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

    @staticmethod
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
                    & (
                        x[i, target_q]
                        ^ z[i, target_q]
                        ^ x[i, control_q]
                        ^ ~z[i, control_q]
                    )
                )
                ^ (
                    x[i, control_q]
                    & (x[i, target_q] ^ x[i, control_q] ^ z[i, control_q])
                )
            )
            z_control_q = x[i, target_q] ^ z[i, target_q] ^ x[i, control_q]
            z_target_q = x[i, target_q] ^ z[i, control_q] ^ x[i, control_q]
            symplectic_matrix[i, nqubits + control_q] = z_control_q
            symplectic_matrix[i, nqubits + target_q] = z_target_q
            tmp = symplectic_matrix[i, control_q]
            symplectic_matrix[i, control_q] = symplectic_matrix[i, target_q]
            symplectic_matrix[i, target_q] = tmp
        return symplectic_matrix

    @staticmethod
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
                ^ (
                    (x[i, target_q] ^ x[i, control_q])
                    & (z[i, target_q] ^ x[i, target_q])
                )
            )
            x_target_q = x[i, control_q] ^ x[i, target_q]
            z_control_q = z[i, control_q] ^ z[i, target_q] ^ x[i, target_q]
            z_target_q = z[i, target_q] ^ x[i, control_q]
            symplectic_matrix[i - 1, target_q] = x_target_q
            symplectic_matrix[i - 1, nqubits + control_q] = z_control_q
            symplectic_matrix[i - 1, nqubits + target_q] = z_target_q
        return symplectic_matrix
