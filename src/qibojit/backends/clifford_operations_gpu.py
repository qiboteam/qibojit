"""Set of custom CuPy operations for the Clifford backend."""

from functools import reduce

import cupy as cp  # pylint: disable=E0401
import cupyx.scipy.sparse as cp_sparse  # pylint: disable=import-error
import numpy
from qibo.backends._clifford_operations import (
    _dim,
    _dim_xz,
    _exponent,
    _get_rxz,
    _packed_size,
)
from scipy import sparse

np = cp

GRIDDIM, BLOCKDIM = 1024, 128


apply_one_qubit_kernel = """
extern "C"
__global__ void apply_{}(unsigned char* symplectic_matrix, const int q, const int qz, const int nrows, const int ncolumns) {{
    _apply_{}(symplectic_matrix, q, qz, nrows, ncolumns);
}}
"""

apply_two_qubits_kernel = """
extern "C"
__global__ void apply_{}(unsigned char* symplectic_matrix, const int control_q, const int target_q, const int cqz, const int tqz, const int nrows, const int ncolumns) {{
    _apply_{}(symplectic_matrix, control_q, target_q, cqz, tqz, nrows, ncolumns);
}}
"""


def one_qubit_kernel_launcher(kernel, symplectic_matrix, q, nqubits):
    qz = nqubits + q
    ncolumns = _dim(nqubits)
    nrows = _packed_size(ncolumns)
    return kernel((GRIDDIM,), (BLOCKDIM,), (symplectic_matrix, q, qz, nrows, ncolumns))


def two_qubits_kernel_launcher(kernel, symplectic_matrix, control_q, target_q, nqubits):
    cqz = nqubits + control_q
    tqz = nqubits + target_q
    ncolumns = _dim(nqubits)
    nrows = _packed_size(ncolumns)
    return kernel(
        (GRIDDIM,),
        (BLOCKDIM,),
        (symplectic_matrix, control_q, target_q, cqz, tqz, nrows, ncolumns),
    )


apply_H = """
__device__ void _apply_H(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (symplectic_matrix[row_idx + q] & symplectic_matrix[row_idx + qz]);
        const unsigned char tmp = symplectic_matrix[row_idx + q];
        symplectic_matrix[row_idx + q] = symplectic_matrix[row_idx + qz];
        symplectic_matrix[row_idx + qz] = tmp;
    };
}
""" + apply_one_qubit_kernel.format("H", "H")

apply_H = cp.RawKernel(apply_H, "apply_H", options=("--std=c++17",))


def H(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_H, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_CNOT = """
__device__ void _apply_CNOT(unsigned char* symplectic_matrix, const int& control_q, const int& target_q, const int& cqz, const int& tqz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + control_q] & symplectic_matrix[row_idx + tqz]
        ) & (symplectic_matrix[row_idx + target_q] ^ ~symplectic_matrix[row_idx + cqz]);
        symplectic_matrix[row_idx + target_q] = (
            symplectic_matrix[row_idx + target_q] ^ symplectic_matrix[row_idx + control_q]
        );
        symplectic_matrix[row_idx + cqz] = (
            symplectic_matrix[row_idx + cqz] ^ symplectic_matrix[row_idx + tqz]
        );
    };
}
""" + apply_two_qubits_kernel.format("CNOT", "CNOT")

apply_CNOT = cp.RawKernel(apply_CNOT, "apply_CNOT", options=("--std=c++17",))


def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CNOT, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_CZ = """
__device__ void _apply_CZ(unsigned char* symplectic_matrix, const int& control_q, const int& target_q, const int& cqz, const int& tqz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (symplectic_matrix[row_idx + target_q] & symplectic_matrix[row_idx + tqz])
            ^ (
                symplectic_matrix[row_idx + control_q]
                & symplectic_matrix[row_idx + target_q]
                & (symplectic_matrix[row_idx + tqz] ^ ~symplectic_matrix[row_idx + cqz])
            )
            ^ (
                symplectic_matrix[row_idx + target_q]
                & (symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + control_q])
            )
        );
        const unsigned char z_control_q = symplectic_matrix[row_idx + target_q] ^ symplectic_matrix[row_idx + cqz];
        const unsigned char z_target_q = symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + control_q];
        symplectic_matrix[row_idx + cqz] = z_control_q;
        symplectic_matrix[row_idx + tqz] = z_target_q;
    };
}
""" + apply_two_qubits_kernel.format("CZ", "CZ")

apply_CZ = cp.RawKernel(apply_CZ, "apply_CZ", options=("--std=c++17",))


def CZ(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CZ, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_S = """
__device__ void _apply_S(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + q] & symplectic_matrix[row_idx + qz]
        );
        symplectic_matrix[row_idx + qz] = symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q];
    };
}
""" + apply_one_qubit_kernel.format("S", "S")

apply_S = cp.RawKernel(apply_S, "apply_S", options=("--std=c++17",))


def S(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_S, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_Z = """
__device__ void _apply_Z(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            (symplectic_matrix[row_idx + q] & symplectic_matrix[row_idx + qz])
            ^ symplectic_matrix[row_idx + q]
            & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
        );
    };
}
""" + apply_one_qubit_kernel.format("Z", "Z")

apply_Z = cp.RawKernel(apply_Z, "apply_Z", options=("--std=c++17",))


def Z(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_Z, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_X = """
__device__ void _apply_X(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (
                symplectic_matrix[row_idx + qz]
                & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
            )
            ^ (symplectic_matrix[row_idx + qz] & symplectic_matrix[row_idx + q])
        );
    };
}
""" + apply_one_qubit_kernel.format("X", "X")

apply_X = cp.RawKernel(apply_X, "apply_X", options=("--std=c++17",))


def X(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_X, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_Y = """
__device__ void _apply_Y(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (
                symplectic_matrix[row_idx + qz]
                & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
            )
            ^ (
                symplectic_matrix[row_idx + q]
                & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
            )
        );
    };
}
""" + apply_one_qubit_kernel.format("Y", "Y")

apply_Y = cp.RawKernel(apply_Y, "apply_Y", options=("--std=c++17",))


def Y(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_Y, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SX = """
__device__ void _apply_SX(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + qz]
            & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
        );
        symplectic_matrix[row_idx + q] = symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q];
    };
}
""" + apply_one_qubit_kernel.format("SX", "SX")

apply_SX = cp.RawKernel(apply_SX, "apply_SX", options=("--std=c++17",))


def SX(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SX, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SDG = """
__device__ void _apply_SDG(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + q]
            & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
        );
        symplectic_matrix[row_idx + qz] = symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q];
    };
}
""" + apply_one_qubit_kernel.format("SDG", "SDG")

apply_SDG = cp.RawKernel(apply_SDG, "apply_SDG", options=("--std=c++17",))


def SDG(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SDG, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SXDG = """
__device__ void _apply_SXDG(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + qz] & symplectic_matrix[row_idx + q]
        );
        symplectic_matrix[row_idx + q] = symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q];
    };
}
""" + apply_one_qubit_kernel.format("SXDG", "SXDG")

apply_SXDG = cp.RawKernel(apply_SXDG, "apply_SXDG", options=("--std=c++17",))


def SXDG(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SXDG, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_RY_pi = """
__device__ void _apply_RY_pi(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + q]
            & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
        );
        const unsigned char zq = symplectic_matrix[row_idx + qz];
        symplectic_matrix[row_idx + qz] = symplectic_matrix[row_idx + q];
        symplectic_matrix[row_idx + q] = zq;
    };
}
""" + apply_one_qubit_kernel.format("RY_pi", "RY_pi")

apply_RY_pi = cp.RawKernel(apply_RY_pi, "apply_RY_pi", options=("--std=c++17",))


def RY_pi(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_RY_pi, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_RY_3pi_2 = """
__device__ void _apply_RY_3pi_2(unsigned char* symplectic_matrix, const int& q, const int& qz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = symplectic_matrix[row_idx + last] ^ (
            symplectic_matrix[row_idx + qz]
            & (symplectic_matrix[row_idx + qz] ^ symplectic_matrix[row_idx + q])
        );
        const unsigned char zq = symplectic_matrix[row_idx + qz];
        symplectic_matrix[row_idx + qz] = symplectic_matrix[row_idx + q];
        symplectic_matrix[row_idx + q] = zq;
    };
}
""" + apply_one_qubit_kernel.format("RY_3pi_2", "RY_3pi_2")

apply_RY_3pi_2 = cp.RawKernel(
    apply_RY_3pi_2, "apply_RY_3pi_2", options=("--std=c++17",)
)


def RY_3pi_2(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_RY_3pi_2, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SWAP = """
__device__ void _apply_SWAP(unsigned char* symplectic_matrix, const int& control_q, const int& target_q, const int& cqz, const int& tqz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (
                symplectic_matrix[row_idx + control_q]
                & symplectic_matrix[row_idx + tqz]
                & (symplectic_matrix[row_idx + target_q] ^ ~symplectic_matrix[row_idx + cqz])
            )
            ^ (
                (symplectic_matrix[row_idx + target_q] ^ symplectic_matrix[row_idx + control_q])
                & (symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + cqz])
                & (symplectic_matrix[row_idx + tqz] ^ ~symplectic_matrix[row_idx + control_q])
            )
            ^ (
                symplectic_matrix[row_idx + target_q]
                & symplectic_matrix[row_idx + cqz]
                & (
                    symplectic_matrix[row_idx + control_q]
                    ^ symplectic_matrix[row_idx + target_q]
                    ^ symplectic_matrix[row_idx + cqz]
                    ^ ~symplectic_matrix[row_idx + tqz]
                )
            )
        );
        const unsigned char x_cq = symplectic_matrix[row_idx + control_q];
        const unsigned char x_tq = symplectic_matrix[row_idx + target_q];
        const unsigned char z_cq = symplectic_matrix[row_idx + cqz];
        const unsigned char z_tq = symplectic_matrix[row_idx + tqz];
        symplectic_matrix[row_idx + control_q] = x_tq;
        symplectic_matrix[row_idx + target_q] = x_cq;
        symplectic_matrix[row_idx + cqz] = z_tq;
        symplectic_matrix[row_idx + tqz] = z_cq;
    };
}
""" + apply_two_qubits_kernel.format("SWAP", "SWAP")

apply_SWAP = cp.RawKernel(apply_SWAP, "apply_SWAP", options=("--std=c++17",))


def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_SWAP, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_iSWAP = """
__device__ void _apply_iSWAP(unsigned char* symplectic_matrix, const int& control_q, const int& target_q, const int& cqz, const int& tqz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (symplectic_matrix[row_idx + target_q] & symplectic_matrix[row_idx + tqz])
            ^ (symplectic_matrix[row_idx + control_q] & symplectic_matrix[row_idx + cqz])
            ^ (
                symplectic_matrix[row_idx + control_q]
                & (symplectic_matrix[row_idx + cqz] ^ symplectic_matrix[row_idx + control_q])
            )
            ^ (
                (symplectic_matrix[row_idx + cqz] ^ symplectic_matrix[row_idx + control_q])
                & (symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + target_q])
                & (symplectic_matrix[row_idx + target_q] ^ ~symplectic_matrix[row_idx + control_q])
            )
            ^ (
                (
                    symplectic_matrix[row_idx + target_q]
                    ^ symplectic_matrix[row_idx + cqz]
                    ^ symplectic_matrix[row_idx + control_q]
                )
                & (
                    symplectic_matrix[row_idx + target_q]
                    ^ symplectic_matrix[row_idx + tqz]
                    ^ symplectic_matrix[row_idx + control_q]
                )
                & (
                    symplectic_matrix[row_idx + target_q]
                    ^ symplectic_matrix[row_idx + tqz]
                    ^ symplectic_matrix[row_idx + control_q]
                    ^ ~symplectic_matrix[row_idx + cqz]
                )
            )
            ^ (
                symplectic_matrix[row_idx + control_q]
                & (
                    symplectic_matrix[row_idx + target_q]
                    ^ symplectic_matrix[row_idx + control_q]
                    ^ symplectic_matrix[row_idx + cqz]
                )
            )
        );
        const unsigned char z_control_q = (
            symplectic_matrix[row_idx + target_q]
            ^ symplectic_matrix[row_idx + tqz]
            ^ symplectic_matrix[row_idx + control_q]
        );
        const unsigned char z_target_q = (
            symplectic_matrix[row_idx + target_q]
            ^ symplectic_matrix[row_idx + cqz]
            ^ symplectic_matrix[row_idx + control_q]
        );
        symplectic_matrix[row_idx + cqz] = z_control_q;
        symplectic_matrix[row_idx + tqz] = z_target_q;
        const unsigned char tmp = symplectic_matrix[row_idx + control_q];
        symplectic_matrix[row_idx + control_q] = symplectic_matrix[row_idx + target_q];
        symplectic_matrix[row_idx + target_q] = tmp;
    };
}
""" + apply_two_qubits_kernel.format("iSWAP", "iSWAP")

apply_iSWAP = cp.RawKernel(apply_iSWAP, "apply_iSWAP", options=("--std=c++17",))


def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_iSWAP, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_CY = """
__device__ void _apply_CY(unsigned char* symplectic_matrix, const int& control_q, const int& target_q, const int& cqz, const int& tqz, const int& nrows, const int& ncolumns) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = ncolumns - 1;
    for(int i = tid; i < nrows; i += ntid) {
        unsigned int row_idx = i * ncolumns;
        symplectic_matrix[row_idx + last] = (
            symplectic_matrix[row_idx + last]
            ^ (
                symplectic_matrix[row_idx + target_q]
                & (symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + target_q])
            )
            ^ (
                symplectic_matrix[row_idx + control_q]
                & (symplectic_matrix[row_idx + target_q] ^ symplectic_matrix[row_idx + tqz])
                & (symplectic_matrix[row_idx + cqz] ^ ~symplectic_matrix[row_idx + target_q])
            )
            ^ (
                (symplectic_matrix[row_idx + target_q] ^ symplectic_matrix[row_idx + control_q])
                & (symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + target_q])
            )
        );
        const unsigned char x_target_q = symplectic_matrix[row_idx + control_q] ^ symplectic_matrix[row_idx + target_q];
        const unsigned char z_control_q = (
            symplectic_matrix[row_idx + cqz]
            ^ symplectic_matrix[row_idx + tqz]
            ^ symplectic_matrix[row_idx + target_q]
        );
        const unsigned char z_target_q = symplectic_matrix[row_idx + tqz] ^ symplectic_matrix[row_idx + control_q];
        symplectic_matrix[row_idx + target_q] = x_target_q;
        symplectic_matrix[row_idx + cqz] = z_control_q;
        symplectic_matrix[row_idx + tqz] = z_target_q;
    };
}
""" + apply_two_qubits_kernel.format("CY", "CY")

apply_CY = cp.RawKernel(apply_CY, "apply_CY", options=("--std=c++17",))


def CY(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CY, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


def _rowsum(symplectic_matrix, h, i, nqubits, determined=False):
    """Updates the symplectic matrix by setting the h-th generator(s) equal to
    the product of the h-th and i-th ones, tracking the resulting phase.

    Ported from the numpy/numba reference (``_clifford_operations._rowsum``):
    the phase-tracking exponent requires genuine (non-bitwise) arithmetic
    difference and sum over the unpacked qubit bits, which cannot be done
    directly on a bit-packed representation, so that part runs unpacked; the
    X/Z updates are then computed with the bit-packed helpers already defined
    in this module, exactly as done there.
    """
    xi, zi = symplectic_matrix[i, :nqubits], symplectic_matrix[i, nqubits:-1]
    xh, zh = symplectic_matrix[h, :nqubits], symplectic_matrix[h, nqubits:-1]
    exponents = _exponent(xi, zi, xh, zh)
    ind = (
        2 * symplectic_matrix[h, -1]
        + 2 * symplectic_matrix[i, -1]
        + cp.sum(exponents, axis=-1)
    ) % 4 == 0
    r = cp.ones(h.shape[0], dtype=cp.uint8)
    r[ind] = 0

    symplectic_matrix = _pack_for_measurements(symplectic_matrix, nqubits)
    packed_n = _packed_size(nqubits)
    xi, zi = symplectic_matrix[i, :packed_n], symplectic_matrix[i, packed_n:-1]
    xh, zh = symplectic_matrix[h, :packed_n], symplectic_matrix[h, packed_n:-1]
    xi_xh = xi ^ xh
    zi_zh = zi ^ zh
    if determined:
        r = reduce(cp.logical_xor, r)
        xi_xh = reduce(cp.logical_xor, xi_xh)
        zi_zh = reduce(cp.logical_xor, zi_zh)
        symplectic_matrix[h[0], -1] = r
        symplectic_matrix[h[0], :packed_n] = xi_xh
        symplectic_matrix[h[0], packed_n:-1] = zi_zh
    else:
        symplectic_matrix[h, -1] = r
        symplectic_matrix[h, :packed_n] = xi_xh
        symplectic_matrix[h, packed_n:-1] = zi_zh
    return _unpack_for_measurements(symplectic_matrix, nqubits)


def _random_outcome(state, p, q, nqubits):
    p = p[0] + nqubits
    tmp = state[p, q].copy()
    state[p, q] = 0
    h = state[:-1, q].nonzero()[0]
    state[p, q] = tmp
    if h.shape[0] > 0:
        state = _rowsum(
            state,
            h.astype(cp.uint),
            p.astype(cp.uint) * cp.ones(h.shape[0], dtype=np.uint),
            nqubits,
            False,
        )
    state[p - nqubits, :] = state[p, :]
    outcome = cp.random.randint(2, size=None, dtype=cp.uint)
    state[p, :] = 0
    state[p, -1] = outcome.astype(cp.uint8)
    state[p, nqubits + q] = 1
    return state, outcome


def _determined_outcome(state, q, nqubits):
    state[-1, :] = 0
    idx = (state[:nqubits, q].nonzero()[0] + nqubits).astype(np.uint)
    if len(idx) == 0:
        return state, state[-1, -1]
    state = _rowsum(
        state,
        (_dim_xz(nqubits) * cp.ones(idx.shape, dtype=np.uint)).astype(np.uint),
        idx.astype(np.uint),
        nqubits,
        True,
    )
    return state, state[-1, -1]


def _packbits(array, axis):
    # cupy.packbits doesn't support axis yet
    return cp.array(numpy.packbits(array.get(), axis=axis), dtype=cp.uint8)


def _unpackbits(array, axis, count):
    return cp.array(
        numpy.unpackbits(array.get(), axis=axis, count=count), dtype=cp.uint8
    )


def _pack_for_measurements(state, nqubits):
    r, x, z = _get_rxz(state, nqubits)
    x = _packbits(x, axis=1)
    z = _packbits(z, axis=1)
    return np.hstack((x, z, r[:, None]))


def _unpack_for_measurements(state, nqubits):
    # unpack the X and Z blocks separately (not as one combined chunk):
    # each was padded independently by _packbits when nqubits isn't a
    # multiple of 8, so unpacking them together misaligns Z by the size of
    # X's padding. Matches _clifford_operations._unpack_for_measurements.
    packed_n = _packed_size(nqubits)
    x = _unpackbits(state[:, :packed_n], axis=1, count=nqubits)
    z = _unpackbits(state[:, packed_n:-1], axis=1, count=nqubits)
    return np.hstack((x, z, state[:, -1][:, None]))


def _init_state_for_measurements(state, nqubits, collapse):
    dim = _dim(nqubits)
    if collapse:
        return _unpackbits(state[None, :], axis=0, count=_dim_xz(nqubits))[:dim]
    else:
        return state.copy()


def cast(x, dtype=None, copy=False):
    if dtype is None:
        dtype = "complex128"

    if cp_sparse.issparse(x):
        if dtype != x.dtype:
            return x.astype(dtype)
        return x

    if sparse.issparse(x):
        cls = getattr(cp_sparse, x.__class__.__name__)
        return cls(x, dtype=dtype)

    if isinstance(x, cp.ndarray) and copy:
        return cp.copy(cp.asarray(x, dtype=dtype))

    return cp.asarray(x, dtype=dtype)


def _clifford_pre_execution_reshape(state):
    return _packbits(state, axis=0).ravel()


def _clifford_post_execution_reshape(state, nqubits):
    dim = _dim(nqubits)
    return _unpackbits(state.reshape(-1, dim), axis=0, count=dim)[:dim]


def maximally_mixed_state(nqubits, dtype=None):
    if dtype is None:
        dtype = "complex128"

    n = 1 << nqubits
    state = cp.eye(n, dtype=dtype)
    cp.cuda.stream.get_current_stream().synchronize()
    state /= 2**nqubits
    return state.reshape((n, n))


def csr_matrix(array, **kwargs):
    return cp_sparse.csr_matrix(array, **kwargs)


def _identity_sparse(dims, dtype=None, **kwargs):
    if dtype is None:  # pragma: no cover
        dtype = "complex128"

    sparsity_format = kwargs.get("format", "csr")

    return cp_sparse.eye(dims, dtype=dtype, format=sparsity_format, **kwargs)
