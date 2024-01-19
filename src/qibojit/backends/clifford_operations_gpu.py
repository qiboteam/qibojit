from functools import cache

import cupy as cp
import numpy as np

GRIDDIM, BLOCKDIM = 1024, 128
BLOCKDIM_2D = (64, 16)
GRIDDIM_2D = (32, 8)

apply_one_qubit_kernel = """
extern "C"
__global__ void apply_{}(bool* symplectic_matrix, const int q, const int nqubits, const int qz, const int dim) {{
    _apply_{}(symplectic_matrix, q, nqubits, qz, dim);
}}
"""

apply_two_qubits_kernel = """
extern "C"
__global__ void apply_{}(bool* symplectic_matrix, const int control_q, const int target_q, const int nqubits, const int cqz, const int tqz, const int dim) {{
    _apply_{}(symplectic_matrix, control_q, target_q, nqubits, cqz, tqz, dim);
}}
"""


def one_qubit_kernel_launcher(kernel, symplectic_matrix, q, nqubits):
    qz = nqubits + q
    dim = 2 * nqubits + 1
    return kernel((GRIDDIM,), (BLOCKDIM,), (symplectic_matrix, q, nqubits, qz, dim))


def two_qubits_kernel_launcher(kernel, symplectic_matrix, control_q, target_q, nqubits):
    cqz = nqubits + control_q
    tqz = nqubits + target_q
    dim = 2 * nqubits + 1
    return kernel(
        (GRIDDIM,),
        (BLOCKDIM,),
        (symplectic_matrix, control_q, target_q, nqubits, cqz, tqz, dim),
    )


apply_H = """
__device__ void _apply_H(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (symplectic_matrix[i * dim + q] & symplectic_matrix[i * dim + qz]);
        const bool tmp = symplectic_matrix[i * dim + q];
        symplectic_matrix[i * dim + q] = symplectic_matrix[i * dim + qz];
        symplectic_matrix[i * dim + qz] = tmp;
    };
}
""" + apply_one_qubit_kernel.format(
    "H", "H"
)

apply_H = cp.RawKernel(apply_H, "apply_H", options=("--std=c++11",))


def H(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_H, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_CNOT = """
__device__ void _apply_CNOT(bool* symplectic_matrix, const int& control_q, const int& target_q, const int& nqubits, const int& cqz, const int& tqz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + control_q] & symplectic_matrix[i * dim + tqz]
        ) & (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + cqz] ^ 1);
        symplectic_matrix[i * dim + target_q] = (
            symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + control_q]
        );
        symplectic_matrix[i * dim + cqz] = (
            symplectic_matrix[i * dim + cqz] ^ symplectic_matrix[i * dim + tqz]
        );
    };
}
""" + apply_two_qubits_kernel.format(
    "CNOT", "CNOT"
)

apply_CNOT = cp.RawKernel(apply_CNOT, "apply_CNOT", options=("--std=c++11",))


def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CNOT, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_CZ = """
__device__ void _apply_CZ(bool* symplectic_matrix, const int& control_q, const int& target_q, const int& nqubits, const int& cqz, const int& tqz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (symplectic_matrix[i * dim + target_q] & symplectic_matrix[i * dim + tqz])
            ^ (
                symplectic_matrix[i * dim + control_q]
                & symplectic_matrix[i * dim + target_q]
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + cqz] ^ 1)
            )
            ^ (
                symplectic_matrix[i * dim + target_q]
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + control_q])
            )
        );
        const bool z_control_q = symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + cqz];
        const bool z_target_q = symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + control_q];
        symplectic_matrix[i * dim + cqz] = z_control_q;
        symplectic_matrix[i * dim + tqz] = z_target_q;
    };
}
""" + apply_two_qubits_kernel.format(
    "CZ", "CZ"
)

apply_CZ = cp.RawKernel(apply_CZ, "apply_CZ", options=("--std=c++11",))


def CZ(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CZ, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_S = """
__device__ void _apply_S(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + q] & symplectic_matrix[i * dim + qz]
        );
        symplectic_matrix[i * dim + qz] = symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q];
    };
}
""" + apply_one_qubit_kernel.format(
    "S", "S"
)

apply_S = cp.RawKernel(apply_S, "apply_S", options=("--std=c++11",))


def S(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_S, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_Z = """
__device__ void _apply_Z(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            (symplectic_matrix[i * dim + q] & symplectic_matrix[i * dim + qz])
            ^ symplectic_matrix[i * dim + q]
            & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
        );
    };
}
""" + apply_one_qubit_kernel.format(
    "Z", "Z"
)

apply_Z = cp.RawKernel(apply_Z, "apply_Z", options=("--std=c++11",))


def Z(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_Z, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_X = """
__device__ void _apply_X(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (
                symplectic_matrix[i * dim + qz]
                & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
            )
            ^ (symplectic_matrix[i * dim + qz] & symplectic_matrix[i * dim + q])
        );
    };
}
""" + apply_one_qubit_kernel.format(
    "X", "X"
)

apply_X = cp.RawKernel(apply_X, "apply_X", options=("--std=c++11",))


def X(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_X, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_Y = """
__device__ void _apply_Y(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (
                symplectic_matrix[i * dim + qz]
                & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
            )
            ^ (
                symplectic_matrix[i * dim + q]
                & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
            )
        );
    };
}
""" + apply_one_qubit_kernel.format(
    "Y", "Y"
)

apply_Y = cp.RawKernel(apply_Y, "apply_Y", options=("--std=c++11",))


def Y(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_Y, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SX = """
__device__ void _apply_SX(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + qz]
            & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
        );
        symplectic_matrix[i * dim + q] = symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q];
    };
}
""" + apply_one_qubit_kernel.format(
    "SX", "SX"
)

apply_SX = cp.RawKernel(apply_SX, "apply_SX", options=("--std=c++11",))


def SX(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SX, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SDG = """
__device__ void _apply_SDG(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + q]
            & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
        );
        symplectic_matrix[i * dim + qz] = symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q];
    };
}
""" + apply_one_qubit_kernel.format(
    "SDG", "SDG"
)

apply_SDG = cp.RawKernel(apply_SDG, "apply_SDG", options=("--std=c++11",))


def SDG(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SDG, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SXDG = """
__device__ void _apply_SXDG(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + qz] & symplectic_matrix[i * dim + q]
        );
        symplectic_matrix[i * dim + q] = symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q];
    };
}
""" + apply_one_qubit_kernel.format(
    "SXDG", "SXDG"
)

apply_SXDG = cp.RawKernel(apply_SXDG, "apply_SXDG", options=("--std=c++11",))


def SXDG(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_SXDG, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_RY_pi = """
__device__ void _apply_RY_pi(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + q]
            & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
        );
        const bool zq = symplectic_matrix[i * dim + qz];
        symplectic_matrix[i * dim + qz] = symplectic_matrix[i * dim + q];
        symplectic_matrix[i * dim + q] = zq;
    };
}
""" + apply_one_qubit_kernel.format(
    "RY_pi", "RY_pi"
)

apply_RY_pi = cp.RawKernel(apply_RY_pi, "apply_RY_pi", options=("--std=c++11",))


def RY_pi(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_RY_pi, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_RY_3pi_2 = """
__device__ void _apply_RY_3pi_2(bool* symplectic_matrix, const int& q, const int& nqubits, const int& qz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = symplectic_matrix[i * dim + last] ^ (
            symplectic_matrix[i * dim + qz]
            & (symplectic_matrix[i * dim + qz] ^ symplectic_matrix[i * dim + q])
        );
        const bool zq = symplectic_matrix[i * dim + qz];
        symplectic_matrix[i * dim + qz] = symplectic_matrix[i * dim + q];
        symplectic_matrix[i * dim + q] = zq;
    };
}
""" + apply_one_qubit_kernel.format(
    "RY_3pi_2", "RY_3pi_2"
)

apply_RY_3pi_2 = cp.RawKernel(
    apply_RY_3pi_2, "apply_RY_3pi_2", options=("--std=c++11",)
)


def RY_3pi_2(symplectic_matrix, q, nqubits):
    one_qubit_kernel_launcher(apply_RY_3pi_2, symplectic_matrix, q, nqubits)
    return symplectic_matrix


apply_SWAP = """
__device__ void _apply_SWAP(bool* symplectic_matrix, const int& control_q, const int& target_q, const int& nqubits, const int& cqz, const int& tqz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (
                symplectic_matrix[i * dim + control_q]
                & symplectic_matrix[i * dim + tqz]
                & (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + cqz] ^ 1)
            )
            ^ (
                (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + control_q])
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + cqz])
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + control_q] ^ 1)
            )
            ^ (
                symplectic_matrix[i * dim + target_q]
                & symplectic_matrix[i * dim + cqz]
                & (
                    symplectic_matrix[i * dim + control_q]
                    ^ symplectic_matrix[i * dim + target_q]
                    ^ symplectic_matrix[i * dim + cqz]
                    ^ symplectic_matrix[i * dim + tqz] ^ 1
                )
            )
        );
        const bool x_cq = symplectic_matrix[i * dim + control_q];
        const bool x_tq = symplectic_matrix[i * dim + target_q];
        const bool z_cq = symplectic_matrix[i * dim + cqz];
        const bool z_tq = symplectic_matrix[i * dim + tqz];
        symplectic_matrix[i * dim + control_q] = x_tq;
        symplectic_matrix[i * dim + target_q] = x_cq;
        symplectic_matrix[i * dim + cqz] = z_tq;
        symplectic_matrix[i * dim + tqz] = z_cq;
    };
}
""" + apply_two_qubits_kernel.format(
    "SWAP", "SWAP"
)

apply_SWAP = cp.RawKernel(apply_SWAP, "apply_SWAP", options=("--std=c++11",))


def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_SWAP, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_iSWAP = """
__device__ void _apply_iSWAP(bool* symplectic_matrix, const int& control_q, const int& target_q, const int& nqubits, const int& cqz, const int& tqz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (symplectic_matrix[i * dim + target_q] & symplectic_matrix[i * dim + tqz])
            ^ (symplectic_matrix[i * dim + control_q] & symplectic_matrix[i * dim + cqz])
            ^ (
                symplectic_matrix[i * dim + control_q]
                & (symplectic_matrix[i * dim + cqz] ^ symplectic_matrix[i * dim + control_q])
            )
            ^ (
                (symplectic_matrix[i * dim + cqz] ^ symplectic_matrix[i * dim + control_q])
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + target_q])
                & (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + control_q] ^ 1)
            )
            ^ (
                (
                    symplectic_matrix[i * dim + target_q]
                    ^ symplectic_matrix[i * dim + cqz]
                    ^ symplectic_matrix[i * dim + control_q]
                )
                & (
                    symplectic_matrix[i * dim + target_q]
                    ^ symplectic_matrix[i * dim + tqz]
                    ^ symplectic_matrix[i * dim + control_q]
                )
                & (
                    symplectic_matrix[i * dim + target_q]
                    ^ symplectic_matrix[i * dim + tqz]
                    ^ symplectic_matrix[i * dim + control_q]
                    ^ symplectic_matrix[i * dim + cqz] ^ 1
                )
            )
            ^ (
                symplectic_matrix[i * dim + control_q]
                & (
                    symplectic_matrix[i * dim + target_q]
                    ^ symplectic_matrix[i * dim + control_q]
                    ^ symplectic_matrix[i * dim + cqz]
                )
            )
        );
        const bool z_control_q = (
            symplectic_matrix[i * dim + target_q]
            ^ symplectic_matrix[i * dim + tqz]
            ^ symplectic_matrix[i * dim + control_q]
        );
        const bool z_target_q = (
            symplectic_matrix[i * dim + target_q]
            ^ symplectic_matrix[i * dim + cqz]
            ^ symplectic_matrix[i * dim + control_q]
        );
        symplectic_matrix[i * dim + cqz] = z_control_q;
        symplectic_matrix[i * dim + tqz] = z_target_q;
        const bool tmp = symplectic_matrix[i * dim + control_q];
        symplectic_matrix[i * dim + control_q] = symplectic_matrix[i * dim + target_q];
        symplectic_matrix[i * dim + target_q] = tmp;
    };
}
""" + apply_two_qubits_kernel.format(
    "iSWAP", "iSWAP"
)

apply_iSWAP = cp.RawKernel(apply_iSWAP, "apply_iSWAP", options=("--std=c++11",))


def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_iSWAP, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


apply_CY = """
__device__ void _apply_CY(bool* symplectic_matrix, const int& control_q, const int& target_q, const int& nqubits, const int& cqz, const int& tqz, const int& dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntid = gridDim.x * blockDim.x;
    const int last = dim - 1;
    for(int i = tid; i < last; i += ntid) {
        symplectic_matrix[i * dim + last] = (
            symplectic_matrix[i * dim + last]
            ^ (
                symplectic_matrix[i * dim + target_q]
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + target_q])
            )
            ^ (
                symplectic_matrix[i * dim + control_q]
                & (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + tqz])
                & (symplectic_matrix[i * dim + cqz] ^ symplectic_matrix[i * dim + target_q] ^ 1)
            )
            ^ (
                (symplectic_matrix[i * dim + target_q] ^ symplectic_matrix[i * dim + control_q])
                & (symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + target_q])
            )
        );
        const bool x_target_q = symplectic_matrix[i * dim + control_q] ^ symplectic_matrix[i * dim + target_q];
        const bool z_control_q = (
            symplectic_matrix[i * dim + cqz]
            ^ symplectic_matrix[i * dim + tqz]
            ^ symplectic_matrix[i * dim + target_q]
        );
        const bool z_target_q = symplectic_matrix[i * dim + tqz] ^ symplectic_matrix[i * dim + control_q];
        symplectic_matrix[i * dim + target_q] = x_target_q;
        symplectic_matrix[i * dim + cqz] = z_control_q;
        symplectic_matrix[i * dim + tqz] = z_target_q;
    };
}
""" + apply_two_qubits_kernel.format(
    "CY", "CY"
)

apply_CY = cp.RawKernel(apply_CY, "apply_CY", options=("--std=c++11",))


def CY(symplectic_matrix, control_q, target_q, nqubits):
    two_qubits_kernel_launcher(
        apply_CY, symplectic_matrix, control_q, target_q, nqubits
    )
    return symplectic_matrix


# This might be optimized using shared memory and warp reduction
_apply_rowsum = """
__device__ void _apply_rowsum(bool* symplectic_matrix, const long* h, const long* i, const int& nqubits, const int& nrows, long* exp, const int& dim) {
    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ntid_x = gridDim.x * blockDim.x;
    unsigned int ntid_y = gridDim.y * blockDim.y;
    const int last = dim - 1;
    for(int j = tid_y; j < nrows; j += ntid_y) {
        for(int k = tid_x; k < nqubits; k += ntid_x) {
            unsigned int kz = nqubits + k;
            bool x1_eq_z1 = symplectic_matrix[i[j] * dim + k] == symplectic_matrix[i[j] * dim + kz];
            bool x1_eq_0 = symplectic_matrix[i[j] * dim + k] == false;
            if (x1_eq_z1) {
                if (not x1_eq_0) {
                    exp[j] += ((int) symplectic_matrix[h[j] * dim + kz]) -
                        (int) symplectic_matrix[h[j] * dim + k];
                }
            } else {
                if (x1_eq_0) {
                    exp[j] += ((int) symplectic_matrix[h[j] * dim + k]) * (
                        1 - 2 * (int) symplectic_matrix[h[j] * dim + kz]
                    );
                } else {
                    exp[j] += ((int) symplectic_matrix[h[j] * dim + kz]) * (
                        2 * (int) symplectic_matrix[h[j] * dim + k] - 1
                    );
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            symplectic_matrix[h[j] * dim + last] = (
                2 * symplectic_matrix[h[j] * dim + last] + 2 * symplectic_matrix[i[j] * dim + last] + exp[j]
            ) % 4 != 0;
        }
        for(int k = tid_x; k < nqubits; k += ntid_x) {
            unsigned int kz = nqubits + k;
            symplectic_matrix[h[j] * dim + k] = (
                symplectic_matrix[i[j] * dim + k] ^ symplectic_matrix[h[j] * dim + k]
            );
            symplectic_matrix[h[j] * dim + kz] = (
                symplectic_matrix[i[j] * dim + kz] ^ symplectic_matrix[h[j] * dim + kz]
            );
        }
    }
}
"""

apply_rowsum = f"""
{_apply_rowsum}
extern "C"
__global__ void apply_rowsum(bool* symplectic_matrix, const long* h, const long* i, const int nqubits, const int nrows, long* exp, const int dim) {{
    _apply_rowsum(symplectic_matrix, h, i, nqubits, nrows, exp, dim);
}}
"""

apply_rowsum = cp.RawKernel(apply_rowsum, "apply_rowsum", options=("--std=c++11",))


@cache
def _get_dim(nqubits):
    return 2 * nqubits + 1


def _rowsum(symplectic_matrix, h, i, nqubits):
    dim = _get_dim(nqubits)
    nrows = len(h)
    exp = cp.zeros(len(h), dtype=int)
    apply_rowsum(
        GRIDDIM_2D, BLOCKDIM_2D, (symplectic_matrix, h, i, nqubits, nrows, exp, dim)
    )
    return symplectic_matrix


def _get_p(state, q, nqubits):
    dim = _get_dim(nqubits)
    return state.reshape(dim, dim)[nqubits:-1, q].nonzero()[0]


def _random_outcome(state, p, q, nqubits):
    dim = _get_dim(nqubits)
    p = p[0] + nqubits
    idx_pq = p * dim + q
    tmp = state[idx_pq].copy()
    state[idx_pq] = False
    h = state.reshape(dim, dim)[:-1, q].nonzero()[0]
    state[idx_pq] = tmp
    if h.shape[0] > 0:
        state = _rowsum(
            state,
            h,
            p.astype(cp.uint) * cp.ones(h.shape[0], dtype=np.uint),
            nqubits,
        )
    state = state.reshape(dim, dim)
    state[p - nqubits, :] = state[p, :]
    outcome = cp.random.randint(2, size=None, dtype=cp.uint)
    state[p, :] = False
    state[p, -1] = outcome.astype(bool)
    state[p, nqubits + q] = True
    return state.ravel(), outcome


__random_outcome = f"""
#include <stdlib.h>
{_apply_rowsum}
__device__ void random_outcome(bool* state, int& p, int& q, int& nqubits, int& dim, int& outcome) {{
    unsigned int tid_y = blockIdx.y;
    unsigned int ntid_y = gridDim.y;
    for(int j = tid_y; j < dim - 1; j += ntid_y) {{
        if (state[j * dim + q] && j != p) {{
            _apply_rowsum(state, j, p, nqubits, dim)
        }}
    }}
    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j = tid_x * (1 + tid_y); j < dim; j += ntid_y) {{
        state[(p - nqubits) * dim + j] = state[p * dim + j];
        state[p * dim + j] = false;
    }}
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {{
        outcome = rand() % 2;
        state[p * dim + dim] = (bool) outcome;
        state[p * dim + nqubits + q] = true;
    }}
}}
"""


def _determined_outcome(state, q, nqubits):
    dim = _get_dim(nqubits)
    state.reshape(dim, dim)[-1, :] = False
    for i in state.reshape(dim, dim)[:nqubits, q].nonzero()[0]:
        state = _rowsum(
            state,
            cp.array([2 * nqubits], dtype=np.uint),
            cp.array([i + nqubits], dtype=np.uint),
            nqubits,
        )
    return state, state[dim * dim - 1].astype(cp.uint)
