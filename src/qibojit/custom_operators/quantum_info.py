# pylint: disable-all

import numba.types as nbt
import numpy as np
import qibo.quantum_info.quantum_info as qinfo
from numba import njit, prange, void
from numba.np.unsafe.ndarray import to_fixed_tuple
from scipy.linalg import expm

ENGINE = qinfo.ENGINE

SIGNATURES = {
    # "_vectorization_row": (
    #    ["c16[:,::1](c16[:,::1], i8)", "c16[:,::1](c16[:,:,::1], i8)"],
    #    {"parallel": True, "cache": True},
    # ),
    # "_unvectorization_row": (
    #    ["c16[:,:,::1](c16[:,::1], i8)", "c16[:,:,::1](c16[:,:,::1], i8)"],
    #    {"parallel": True, "cache": True},
    # ),
    # "_random_hermitian": ("c16[:,:](i8)", {"parallel": True, "cache": True})
    # "_vectorize_pauli_basis_row": ("c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)", {"parallel": True, "cache": True}),
    # "_vectorize_pauli_basis_column": ("c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)", {"parallel": True, "cache": True}),
}


def jit_function(signature, function, **kwargs):
    if isinstance(function, str):
        function = getattr(qinfo, function)
    return njit(signature, **kwargs)(function)


class QinfoNumba:
    pass


QINFO = QinfoNumba()

for function, signature in SIGNATURES.items():
    print(function)
    jitted = jit_function(signature[0], function, **signature[1])
    globals()[function] = jitted
    setattr(QINFO, function, jitted)


@njit("c16[:,:,::1](i8, c16[:,:,:,::1])", parallel=True, cache=True)
def _pauli_basis_inner(
    nqubits,
    prod,
):
    dim = 2**nqubits
    basis = np.empty((len(prod), dim, dim), dtype=np.complex128)
    for i in prange(len(prod)):  # pylint: disable=not-an-iterable
        elem = prod[i][0]
        for j in prange(1, len(prod[i])):  # pylint: disable=not-an-iterable
            elem = np.kron(elem, prod[i][j])
        basis[i] = elem
    return basis


@njit("c16[:,:,:,::1](c16[:,:,::1], i8)", parallel=False, cache=True)
def _cartesian_product(arrays, n):
    num_arrays = len(arrays)
    num_elements = num_arrays**n  # Total number of combinations
    result = np.empty(
        (num_elements, n, arrays[0].shape[0], arrays[0].shape[1]), dtype=arrays[0].dtype
    )

    # Generate Cartesian product using a lexicographic order
    indices = np.empty(n, dtype=np.int64)

    for i in range(num_elements):
        temp = i
        for j in range(n - 1, -1, -1):  # Iterate right-to-left for lexicographic order
            indices[j] = temp % num_arrays
            temp //= num_arrays

        # Fill the result array with selected elements
        for j in range(n):  # pylint: disable=not-an-iterable
            result[i, j] = arrays[indices[j]]

    return result


@njit(
    "c16[:,:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)",
    parallel=True,
    cache=True,
)
def _pauli_basis(
    nqubits: int,
    pauli_0,
    pauli_1,
    pauli_2,
    pauli_3,
    normalization: float = 1.0,
):
    basis = np.empty((4, 2, 2), dtype=pauli_0.dtype)
    # Assign manually instead of using vstack
    basis[0] = pauli_0
    basis[1] = pauli_1
    basis[2] = pauli_2
    basis[3] = pauli_3
    prod = _cartesian_product(basis, nqubits)
    return _pauli_basis_inner(nqubits, prod) / normalization


setattr(QINFO, "_pauli_basis", _pauli_basis)


@njit(["c16[:,:](c16[:,:], i8[:])", "c16[:,:,:](c16[:,:,:], i8[:])"], cache=True)
def numba_transpose(array, axes):
    axes = to_fixed_tuple(axes, array.ndim)
    array = np.transpose(array, axes)
    # return np.ascontiguousarray(array)
    return array


@njit(
    ["c16[:,::1](c16[:,:], i8)", "c16[:,::1](c16[:,:,:], i8)"],
    parallel=True,
    cache=True,
)
def _vectorization_row(state, dim: int):
    return ENGINE.reshape(ENGINE.ascontiguousarray(state), (-1, dim**2))


setattr(QINFO, "_vectorization_row", _vectorization_row)


@njit(["c16[:,::1](c16[:,:], i8)", "c16[:,::1](c16[:,:,:], i8)"], cache=True)
def _vectorization_column(state, dim):
    indices = ENGINE.arange(state.ndim)
    indices[-2:] = indices[-2:][::-1]
    state = numba_transpose(state, indices)
    state = ENGINE.ascontiguousarray(state)
    return ENGINE.reshape(state, (-1, dim**2))


# dynamic tuple creation is not possible in numba
@njit
def _vectorization_system(state, dim=0):
    nqubits = int(ENGINE.log2(state.shape[-1]))
    new_axis = [
        0,
    ]
    for qubit in range(nqubits):
        new_axis.extend([qubit + nqubits + 1, qubit + 1])
    state = ENGINE.reshape(state, (-1,) + (2,) * 2 * nqubits)
    state = numba_transpose(state, new_axis)
    return ENGINE.reshape(state, (-1, 2 ** (2 * nqubits)))


setattr(QINFO, "_vectorization_column", _vectorization_column)
# setattr(QINFO, "_vectorization_system", _vectorization_system)


@njit(
    ["c16[:,:,::1](c16[:,:], i8)", "c16[:,:,::1](c16[:,:,:], i8)"],
    parallel=True,
    cache=True,
)
def _unvectorization_row(state, dim: int):
    return ENGINE.reshape(ENGINE.ascontiguousarray(state), (state.shape[0], dim, dim))


setattr(QINFO, "_unvectorization_row", _unvectorization_row)


@njit(["c16[:,:,:](c16[:,:], i8)", "c16[:,:,:](c16[:,:,:], i8)"], cache=True)
def _unvectorization_column(state, dim):
    # axes = ENGINE.arange(state.ndim)[::-1]
    last_dim = state.shape[0]
    state = state.T  # numba_transpose(state, axes)
    state = ENGINE.ascontiguousarray(state).reshape(dim, dim, last_dim)
    # return numba_transpose(state, ENGINE.array([2, 1, 0], dtype=ENGINE.int64))
    return state.T


setattr(QINFO, "_unvectorization_column", _unvectorization_column)


@njit(
    [
        nbt.complex128[:](
            nbt.complex128[:, :], nbt.Tuple((nbt.int64[:], nbt.int64[:]))
        ),
        nbt.float64[:](nbt.float64[:, :], nbt.Tuple((nbt.int64[:], nbt.int64[:]))),
        nbt.int64[:](nbt.int64[:, :], nbt.Tuple((nbt.int64[:], nbt.int64[:]))),
    ],
    parallel=True,
    cache=True,
)
def _array_at_2d_indices(array, indices):
    empty = ENGINE.empty(indices[0].shape, dtype=array.dtype)
    for i in prange(len(indices[0])):
        empty[i] = array[indices[0][i], indices[1][i]]
    return empty


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[:, ::1]))(
        nbt.complex128[:, :], nbt.int64
    ),
    cache=True,
)
def _post_sparse_pauli_basis_vectorization(basis, dim):
    indices = ENGINE.nonzero(basis)
    basis = _array_at_2d_indices(basis, indices)
    basis = ENGINE.ascontiguousarray(basis).reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


setattr(
    QINFO,
    "_post_sparse_pauli_basis_vectorization",
    _post_sparse_pauli_basis_vectorization,
)


@njit(
    "c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)",
    parallel=True,
    cache=True,
)
def _vectorize_pauli_basis_row(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    dim = 2**nqubits
    basis = _pauli_basis(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _vectorization_row(basis, dim)


setattr(QINFO, "_vectorize_pauli_basis_row", _vectorize_pauli_basis_row)


@njit(
    "c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)",
    parallel=True,
    cache=True,
)
def _vectorize_pauli_basis_column(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    dim = 2**nqubits
    basis = _pauli_basis(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _vectorization_column(basis, dim)


setattr(QINFO, "_vectorize_pauli_basis_column", _vectorize_pauli_basis_column)


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[:, ::1]))(
        nbt.int64,
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.float64,
    ),
    parallel=True,
    cache=True,
)
def _vectorize_sparse_pauli_basis_row(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    dim = 2**nqubits
    basis = _vectorize_pauli_basis_row(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    return _post_sparse_pauli_basis_vectorization(basis, dim)


setattr(QINFO, "_vectorize_sparse_pauli_basis_row", _vectorize_sparse_pauli_basis_row)


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[:, ::1]))(
        nbt.int64,
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.float64,
    ),
    parallel=True,
    cache=True,
)
def _vectorize_sparse_pauli_basis_column(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    dim = 2**nqubits
    basis = _vectorize_pauli_basis_column(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    return _post_sparse_pauli_basis_vectorization(basis, dim)


setattr(
    QINFO, "_vectorize_sparse_pauli_basis_column", _vectorize_sparse_pauli_basis_column
)


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[:]))(
        nbt.int64,
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.complex128[:, ::1],
        nbt.float64,
    ),
    parallel=True,
    cache=True,
)
def _pauli_to_comp_basis_sparse_row(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    unitary = _vectorize_pauli_basis_row(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    # unitary = numba_transpose(unitary, ENGINE.arange(unitary.ndim)[::-1])
    unitary = unitary.T
    nonzero = ENGINE.nonzero(unitary)
    unitary = _array_at_2d_indices(unitary, nonzero)
    return ENGINE.ascontiguousarray(unitary).reshape(unitary.shape[0], -1), nonzero[1]


setattr(QINFO, "_pauli_to_comp_basis_sparse_row", _pauli_to_comp_basis_sparse_row)


@njit(
    nbt.Tuple((nbt.complex128[:, :], nbt.complex128[:, :], nbt.float64[:, :, ::1]))(
        nbt.complex128[:, :]
    ),
    parallel=True,
    cache=True,
)
def _choi_to_kraus_preamble(choi_super_op):
    U, coefficients, V = ENGINE.linalg.svd(choi_super_op)
    # U = np.ascontiguousarray(U)
    # U = numba_transpose(U, ENGINE.arange(U.ndim)[::-1])
    U = U.T
    coefficients = ENGINE.sqrt(coefficients)
    V = ENGINE.conj(V)
    coefficients = coefficients.reshape(U.shape[0], 1, 1)
    # V = np.ascontiguousarray(V)
    return U, V, coefficients


@njit("c16[:,:,:,:](c16[:,:,:], c16[:,:,:])", parallel=True, cache=True)
def _kraus_operators(kraus_left, kraus_right):
    kraus_ops = ENGINE.empty((2,) + kraus_left.shape, dtype=kraus_left.dtype)
    kraus_ops[0] = kraus_left
    kraus_ops[1] = kraus_right
    return kraus_ops


@njit(
    nbt.Tuple((nbt.complex128[:, :, :, :], nbt.float64[:, :, ::1]))(
        nbt.complex128[:, ::1]
    ),
    cache=True,
)
def _choi_to_kraus_row(choi_super_op):
    U, V, coefficients = _choi_to_kraus_preamble(choi_super_op)
    dim = int(np.sqrt(U.shape[-1]))
    kraus_left = coefficients * _unvectorization_row(U, dim)
    kraus_right = coefficients * _unvectorization_row(V, dim)
    kraus_ops = _kraus_operators(kraus_left, kraus_right)
    return kraus_ops, coefficients


setattr(QINFO, "_choi_to_kraus_row", _choi_to_kraus_row)


@njit(
    nbt.Tuple((nbt.complex128[:, :, :, :], nbt.float64[:, :, ::1]))(
        nbt.complex128[:, ::1]
    ),
    cache=True,
)
def _choi_to_kraus_column(choi_super_op):
    U, V, coefficients = _choi_to_kraus_preamble(choi_super_op)
    dim = int(np.sqrt(U.shape[-1]))
    kraus_left = coefficients * _unvectorization_column(U, dim)
    kraus_right = coefficients * _unvectorization_column(V, dim)
    kraus_ops = _kraus_operators(kraus_left, kraus_right)
    return kraus_ops, coefficients


setattr(QINFO, "_choi_to_kraus_column", _choi_to_kraus_column)


@njit("c16[:](i8)", parallel=True, cache=True)
def _random_statevector(dims: int):
    state = ENGINE.random.standard_normal(dims)
    state = state + 1.0j * ENGINE.random.standard_normal(dims)
    return state / ENGINE.linalg.norm(state)


setattr(QINFO, "_random_statevector", _random_statevector)


@njit("c16[:,:](i8, i8, f8, f8)", parallel=True, cache=True)
def _random_gaussian_matrix(dims: int, rank: int, mean: float, stddev: float):
    matrix = ENGINE.empty((dims, rank), dtype=ENGINE.complex128)
    for i in prange(dims):
        for j in prange(rank):
            matrix[i, j] = ENGINE.random.normal(
                loc=mean, scale=stddev
            ) + 1.0j * ENGINE.random.normal(loc=mean, scale=stddev)
    return matrix


setattr(QINFO, "_random_gaussian_matrix", _random_gaussian_matrix)


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_density_matrix_pure(dims: int):
    state = _random_statevector(dims)
    return ENGINE.outer(state, ENGINE.conj(state).T)


setattr(QINFO, "_random_density_matrix_pure", _random_density_matrix_pure)


@njit("c16[:,:](i8, i8, f8, f8)", parallel=True, cache=True)
def _random_density_matrix_hs_ginibre(dims: int, rank: int, mean: float, stddev: float):
    state = _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.transpose(ENGINE.conj(state), (1, 0))
    return state / ENGINE.trace(state)


setattr(QINFO, "_random_density_matrix_hs_ginibre", _random_density_matrix_hs_ginibre)


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_hermitian(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return (matrix + ENGINE.conj(matrix).T) / 2


setattr(QINFO, "_random_hermitian", _random_hermitian)


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_hermitian_semidefinite(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return ENGINE.conj(matrix).T @ matrix


setattr(QINFO, "_random_hermitian_semidefinite", _random_hermitian_semidefinite)


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_unitary_haar(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    Q, R = ENGINE.linalg.qr(matrix)
    D = ENGINE.diag(R)
    D = D / ENGINE.abs(D)
    R = ENGINE.diag(D)
    return ENGINE.ascontiguousarray(Q) @ R


setattr(QINFO, "_random_unitary_haar", _random_unitary_haar)

"""
# double check whether this is correct
#@njit
def expm(A):
    '''Compute expm(A) using the Padé approximant and scaling/squaring.'''
    # Constants for Padé approximant
    pade_coeffs = np.array([
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0
    ])

    n = A.shape[0]
    A_norm = np.max(np.sum(np.abs(A), axis=1))  # Compute norm estimate

    # Scaling step
    s = max(0, int(np.log2(A_norm)) - 4)
    A_scaled = A / (2 ** s)

    # Compute Padé approximant
    X = A_scaled @ A_scaled
    U = ENGINE.eye(n, dtype=A.dtype) * pade_coeffs[1]
    V = ENGINE.eye(n, dtype=A.dtype) * pade_coeffs[0]

    for i in range(2, len(pade_coeffs)):
        U = X @ U + pade_coeffs[i] * ENGINE.eye(n, dtype=A.dtype)
        V = X @ V + pade_coeffs[i - 1] * ENGINE.eye(n, dtype=A.dtype)

    U = A_scaled @ U
    P = V + U
    Q = V - U

    breakpoint()
    # Solve (I - U)⁻¹ * (I + U)
    F = ENGINE.linalg.solve(Q, P)

    # Squaring step
    for _ in range(s):
        F = F @ F

    return F
"""


def _random_unitary(dims: int):
    H = _random_hermitian(dims)
    return expm(-1.0j * H / 2)


setattr(QINFO, "_random_unitary", _random_unitary)


@njit("c16[:,:](c16[:,:], i8, i8, f8, f8)", parallel=True, cache=True)
def _random_density_matrix_bures_inner(
    unitary, dims: int, rank: int, mean: float, stddev: float
):
    state = ENGINE.eye(dims, dtype=unitary.dtype)
    state += unitary
    state = state @ _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.transpose(ENGINE.conj(state), (1, 0))
    return state / ENGINE.trace(state)


def _random_density_matrix_bures(dims: int, rank: int, mean: float, stddev: float):
    unitary = _random_unitary(dims)
    return _random_density_matrix_bures_inner(unitary, dims, rank, mean, stddev)


setattr(QINFO, "_random_density_matrix_bures", _random_density_matrix_bures)


@njit(nbt.Tuple((nbt.int64[:], nbt.int64[:]))(nbt.int64), parallel=True, cache=True)
def _sample_from_quantum_mallows_distribution(nqubits: int):
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64)
    powers = 4**exponents
    powers[powers == 0] = ENGINE.iinfo(ENGINE.int64).max
    r = ENGINE.random.uniform(0, 1, size=nqubits)
    indexes = (-1) * ENGINE.ceil(ENGINE.log2(r + (1 - r) / powers)).astype(ENGINE.int64)
    idx_le_exp = indexes < exponents
    hadamards = idx_le_exp.astype(ENGINE.int64)
    idx_gt_exp = idx_le_exp ^ True
    indexes[idx_gt_exp] = 2 * exponents[idx_gt_exp] - indexes[idx_gt_exp] - 1
    mute_index = list(range(nqubits))
    permutations = ENGINE.zeros(nqubits, dtype=ENGINE.int64)
    for l, index in enumerate(indexes):
        permutations[l] = mute_index[index]
        del mute_index[index]
    return hadamards, permutations


setattr(
    QINFO,
    "_sample_from_quantum_mallows_distribution",
    _sample_from_quantum_mallows_distribution,
)


@njit(
    [
        void(
            nbt.complex128[:, :],
            nbt.Tuple((nbt.int64[:], nbt.int64[:])),
            nbt.complex128[:],
        ),
        void(
            nbt.float64[:, :], nbt.Tuple((nbt.int64[:], nbt.int64[:])), nbt.float64[:]
        ),
        void(nbt.int64[:, :], nbt.Tuple((nbt.int64[:], nbt.int64[:])), nbt.int64[:]),
    ],
    parallel=True,
    cache=True,
)
def _set_array_at_2d_indices(array, indices, values):
    for i in prange(len(indices[0])):
        array[indices[0][i], indices[1][i]] = values[i]


@njit(
    nbt.Tuple((nbt.int64[:, :], nbt.int64[:, :], nbt.int64[:, :], nbt.int64[:, :]))(
        nbt.int64, nbt.int64[:], nbt.int64[:]
    ),
    parallel=True,
    cache=True,
)
def _gamma_delta_matrices(nqubits: int, hadamards, permutations):
    delta_matrix = ENGINE.eye(nqubits, dtype=ENGINE.int64)
    delta_matrix_prime = ENGINE.copy(delta_matrix)

    gamma_matrix_prime = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix_prime = ENGINE.diag(gamma_matrix_prime)

    gamma_matrix = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = ENGINE.diag(gamma_matrix)

    tril_indices = ENGINE.tril_indices(nqubits, k=-1)
    _set_array_at_2d_indices(
        delta_matrix_prime,
        tril_indices,
        ENGINE.random.randint(0, 2, size=len(tril_indices[0])),
    )

    _set_array_at_2d_indices(
        gamma_matrix_prime,
        tril_indices,
        ENGINE.random.randint(0, 2, size=len(tril_indices[0])),
    )

    triu_indices = ENGINE.triu_indices(nqubits, k=1)
    _set_array_at_2d_indices(
        gamma_matrix_prime,
        triu_indices,
        _array_at_2d_indices(gamma_matrix_prime, tril_indices),
    )

    p_col_gt_row = permutations[triu_indices[1]] > permutations[triu_indices[0]]
    p_col_neq_row = permutations[triu_indices[1]] != permutations[triu_indices[0]]
    p_col_le_row = p_col_gt_row ^ True
    h_row_eq_0 = hadamards[triu_indices[0]] == 0
    h_col_eq_0 = hadamards[triu_indices[1]] == 0

    idx = (h_row_eq_0 * h_col_eq_0 ^ True) * p_col_neq_row
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0]))
    _set_array_at_2d_indices(
        gamma_matrix, (triu_indices[0][idx], triu_indices[1][idx]), elements
    )
    _set_array_at_2d_indices(
        gamma_matrix, (triu_indices[1][idx], triu_indices[0][idx]), elements
    )

    idx = p_col_gt_row | (p_col_le_row * h_row_eq_0 * h_col_eq_0)
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0]))
    _set_array_at_2d_indices(
        delta_matrix, (triu_indices[1][idx], triu_indices[0][idx]), elements
    )

    return gamma_matrix, gamma_matrix_prime, delta_matrix, delta_matrix_prime


setattr(QINFO, "_gamma_delta_matrices", _gamma_delta_matrices)


"""
@njit(
    [
        nbt.complex128[::1](
            nbt.complex128[:, ::1], nbt.Tuple((nbt.int64[::1], nbt.int64[::1]))
        ),
        nbt.float64[::1](
            nbt.float64[:, ::1], nbt.Tuple((nbt.int64[::1], nbt.int64[::1]))
        ),
    ],
    parallel=True,
    cache=True,
)
def _set_array_at_2d_indices(array, indices, values):
    empty = ENGINE.empty(indices[0].shape, dtype=array.dtype)
    for i in prange(len(indices[0])):
        empty[i] = array[indices[0][i], indices[1][i]]
    return empty
"""

"""
def _kraus_to_stinespring(
    kraus_ops, initial_state_env, dim_env: int
):
    alphas = ENGINE.zeros((dim_env, dim_env, dim_env), dtype=complex)
    #idx = ENGINE.arange(dim_env)
    alphas[range(dim_env), range(dim_env)] = initial_state_env
    # batched kron product
    prod = 0.
    for i in prange(len(kraus_ops)):
        prod += ENGINE.kron(kraus_ops[i], alphas[i])
    return prod.reshape(
        2 * (kraus_ops.shape[1] * alphas.shape[1],)
    )
"""

# setattr(QINFO, "_kraus_to_stinespring", _kraus_to_stinespring)
