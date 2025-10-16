# pylint: disable-all

import numba.types as nbt
import qibo.quantum_info._quantum_info as qinfo
from numba import njit, prange, void
from numba.np.unsafe.ndarray import to_fixed_tuple
from scipy.linalg import expm

ENGINE = qinfo.ENGINE  # this should be numpy


@njit("(i8,)", cache=True)
def set_seed(seed):
    ENGINE.random.seed(seed)


@njit("c16[:,:,::1](i8, c16[:,:,:,::1])", parallel=True, cache=True)
def _pauli_basis_inner(
    nqubits,
    prod,
):
    dim = 2**nqubits
    basis = ENGINE.empty((len(prod), dim, dim), dtype=ENGINE.complex128)
    for i in prange(len(prod)):
        elem = prod[i][0]
        for j in prange(1, len(prod[i])):
            elem = ENGINE.kron(elem, prod[i][j])
        basis[i] = elem
    return basis


@njit("c16[:,:,:,::1](c16[:,:,::1], i8)", parallel=False, cache=True)
def _cartesian_product(arrays, n):
    num_arrays = len(arrays)
    num_elements = num_arrays**n  # Total number of combinations
    result = ENGINE.empty(
        (num_elements, n, arrays[0].shape[0], arrays[0].shape[1]), dtype=arrays[0].dtype
    )

    # Generate Cartesian product using a lexicographic order
    indices = ENGINE.empty(n, dtype=ENGINE.int64)

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
    basis = ENGINE.empty((4, 2, 2), dtype=pauli_0.dtype)
    # Assign manually instead of using vstack
    basis[0] = pauli_0
    basis[1] = pauli_1
    basis[2] = pauli_2
    basis[3] = pauli_3
    prod = _cartesian_product(basis, nqubits)
    return _pauli_basis_inner(nqubits, prod) / normalization


@njit(["c16[:,:](c16[:,:], i8[:])", "c16[:,:,:](c16[:,:,:], i8[:])"], cache=True)
def numba_transpose(array, axes):
    axes = to_fixed_tuple(axes, array.ndim)
    array = ENGINE.transpose(array, axes)
    return array


@njit(
    ["c16[:,::1](c16[:,:], i8)", "c16[:,::1](c16[:,:,:], i8)"],
    parallel=True,
    cache=True,
)
def _vectorization_row(state, dim: int):
    return ENGINE.reshape(ENGINE.ascontiguousarray(state), (-1, dim**2))


@njit(["c16[:,::1](c16[:,:], i8)", "c16[:,::1](c16[:,:,:], i8)"], cache=True)
def _vectorization_column(state, dim):
    indices = ENGINE.arange(state.ndim)
    indices[-2:] = indices[-2:][::-1]
    state = ENGINE.transpose(state, to_fixed_tuple(indices, state.ndim))
    state = ENGINE.ascontiguousarray(state)
    return ENGINE.reshape(state, (-1, dim**2))


# dynamic tuple creation is not possible in numba
# this might be jittable if we passed the shape
# dim = (2,) * 2 * nqubits as inputs
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


@njit(
    ["c16[:,:,::1](c16[:,:], i8)", "c16[:,:,::1](c16[:,:,:], i8)"],
    parallel=True,
    cache=True,
)
def _unvectorization_row(state, dim: int):
    return ENGINE.reshape(ENGINE.ascontiguousarray(state), (state.shape[0], dim, dim))


@njit(["c16[:,:,:](c16[:,:], i8)", "c16[:,:,:](c16[:,:,:], i8)"], cache=True)
def _unvectorization_column(state, dim):
    # axes = ENGINE.arange(state.ndim)[::-1]
    last_dim = state.shape[0]
    state = state.T  # numba_transpose(state, axes)
    state = ENGINE.ascontiguousarray(state).reshape(dim, dim, last_dim)
    # return numba_transpose(state, ENGINE.array([2, 1, 0], dtype=ENGINE.int64))
    return state.T


@njit("c16[:,:](c16[:,:], i8, i8)", parallel=True, cache=True)
def _reshuffling(super_op, ax1: int, ax2: int):
    dim = int(ENGINE.sqrt(super_op.shape[0]))
    super_op = ENGINE.reshape(ENGINE.ascontiguousarray(super_op), (dim, dim, dim, dim))
    axes = ENGINE.arange(super_op.ndim)
    tmp = axes[ax1]
    axes[ax1] = axes[ax2]
    axes[ax2] = tmp
    # axes[[ax1, ax2]] = axes[[ax2, ax1]]
    super_op = ENGINE.transpose(super_op, to_fixed_tuple(axes, 4))
    return ENGINE.reshape(ENGINE.ascontiguousarray(super_op), (dim**2, dim**2))


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
def _pauli_to_comp_basis_sparse_column(
    nqubits: int, pauli_0, pauli_1, pauli_2, pauli_3, normalization: float = 1.0
):
    unitary = _vectorize_pauli_basis_column(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    unitary = unitary.T
    nonzero = ENGINE.nonzero(unitary)
    unitary = _array_at_2d_indices(unitary, nonzero)
    return ENGINE.ascontiguousarray(unitary).reshape(unitary.shape[0], -1), nonzero[1]


@njit(
    nbt.Tuple((nbt.complex128[:, :], nbt.complex128[:, :], nbt.float64[:, :, ::1]))(
        nbt.complex128[:, :]
    ),
    parallel=True,
    cache=True,
)
def _choi_to_kraus_preamble(choi_super_op):
    U, coefficients, V = ENGINE.linalg.svd(choi_super_op)
    # U = ENGINE.ascontiguousarray(U)
    # U = numba_transpose(U, ENGINE.arange(U.ndim)[::-1])
    U = U.T
    coefficients = ENGINE.sqrt(coefficients)
    V = ENGINE.conj(V)
    coefficients = coefficients.reshape(U.shape[0], 1, 1)
    # V = ENGINE.ascontiguousarray(V)
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
    dim = int(ENGINE.sqrt(U.shape[-1]))
    kraus_left = coefficients * _unvectorization_row(U, dim)
    kraus_right = coefficients * _unvectorization_row(V, dim)
    kraus_ops = _kraus_operators(kraus_left, kraus_right)
    return kraus_ops, coefficients


@njit(
    nbt.Tuple((nbt.complex128[:, :, :, :], nbt.float64[:, :, ::1]))(
        nbt.complex128[:, ::1]
    ),
    cache=True,
)
def _choi_to_kraus_column(choi_super_op):
    U, V, coefficients = _choi_to_kraus_preamble(choi_super_op)
    dim = int(ENGINE.sqrt(U.shape[-1]))
    kraus_left = coefficients * _unvectorization_column(U, dim)
    kraus_right = coefficients * _unvectorization_column(V, dim)
    kraus_ops = _kraus_operators(kraus_left, kraus_right)
    return kraus_ops, coefficients


@njit("c16[:](i8)", parallel=True, cache=True)
def _random_statevector(dims: int):
    state = ENGINE.random.standard_normal(dims)
    state = state + 1.0j * ENGINE.random.standard_normal(dims)
    return state / ENGINE.linalg.norm(state)


@njit("c16[:,:](i8, i8, f8, f8)", parallel=True, cache=True)
def _random_gaussian_matrix(dims: int, rank: int, mean: float, stddev: float):
    matrix = ENGINE.empty((dims, rank), dtype=ENGINE.complex128)
    for i in prange(dims):
        for j in prange(rank):
            matrix[i, j] = ENGINE.random.normal(
                loc=mean, scale=stddev
            ) + 1.0j * ENGINE.random.normal(loc=mean, scale=stddev)
    return matrix


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_density_matrix_pure(dims: int):
    state = _random_statevector(dims)
    return ENGINE.outer(state, ENGINE.conj(state).T)


@njit("c16[:,:](i8, i8, f8, f8)", parallel=True, cache=True)
def _random_density_matrix_hs_ginibre(dims: int, rank: int, mean: float, stddev: float):
    state = _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.transpose(ENGINE.conj(state), (1, 0))
    return state / ENGINE.trace(state)


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_hermitian(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return (matrix + ENGINE.conj(matrix).T) / 2


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_hermitian_semidefinite(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return ENGINE.conj(matrix).T @ matrix


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _random_unitary_haar(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    Q, R = ENGINE.linalg.qr(matrix)
    D = ENGINE.diag(R)
    D = D / ENGINE.abs(D)
    R = ENGINE.diag(D)
    return ENGINE.ascontiguousarray(Q) @ R


"""
# double check whether this is correct
#@njit
def expm(A):
    '''Compute expm(A) using the Padé approximant and scaling/squaring.'''
    # Constants for Padé approximant
    pade_coeffs = ENGINE.array([
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0
    ])

    n = A.shape[0]
    A_norm = ENGINE.max(ENGINE.sum(ENGINE.abs(A), axis=1))  # Compute norm estimate

    # Scaling step
    s = max(0, int(ENGINE.log2(A_norm)) - 4)
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


# if we can implement the expm in pure numba
# we will be able to completely jit random unitary
# and the other functions that depend on it
def _random_unitary(dims: int):
    H = _random_hermitian(dims)
    return expm(-1.0j * H / 2)


@njit("c16[:,:](c16[:,:], i8, i8, f8, f8)", parallel=True, cache=True)
def _random_density_matrix_bures_inner(
    unitary, dims: int, rank: int, mean: float, stddev: float
):
    state = ENGINE.eye(dims, dtype=unitary.dtype)
    state += unitary
    state = state @ _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.transpose(ENGINE.conj(state), (1, 0))
    return state / ENGINE.trace(state)


# not entirely jittable because depends on random unitary
def _random_density_matrix_bures(dims: int, rank: int, mean: float, stddev: float):
    unitary = _random_unitary(dims)
    return _random_density_matrix_bures_inner(unitary, dims, rank, mean, stddev)


@njit(nbt.Tuple((nbt.int64[:], nbt.int64[:]))(nbt.int64), parallel=False, cache=True)
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
    permutations = ENGINE.empty(nqubits, dtype=ENGINE.int64)
    mask = ENGINE.ones(nqubits, dtype=ENGINE.bool)
    for l, index in enumerate(indexes):
        available = ENGINE.flatnonzero(mask)
        permutations[l] = available[index]
        mask[permutations[l]] = False
    return hadamards, permutations


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


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.complex128[:, :]))(nbt.int64, nbt.int64),
    parallel=True,
    cache=True,
)
def _super_op_from_bcsz_measure_preamble(dims: int, rank: int):
    super_op = _random_gaussian_matrix(
        dims**2,
        rank=rank,
        mean=0,
        stddev=1,
    )
    super_op = super_op @ ENGINE.conj(super_op).T
    # partial trace
    super_op_reshaped = ENGINE.reshape(super_op, (dims, dims, dims, dims))
    super_op_reduced = ENGINE.empty((dims, dims), dtype=super_op.dtype)
    for j in prange(dims):
        for k in prange(dims):
            super_op_reduced[j, k] = ENGINE.sum(super_op_reshaped[:, j, :, k])
    eigenvalues, eigenvectors = ENGINE.linalg.eigh(super_op_reduced)
    eigenvalues = ENGINE.sqrt(1.0 / eigenvalues)
    eigenvectors = eigenvectors.T
    conj_eigenvectors = ENGINE.conj(eigenvectors)
    operator = ENGINE.zeros(
        (eigenvectors.shape[1], eigenvectors.shape[1]), dtype=eigenvectors.dtype
    )
    for i in prange(eigenvectors.shape[0]):
        operator += eigenvalues[i] * ENGINE.outer(eigenvectors[i], conj_eigenvectors[i])
    return operator, super_op


@njit("c16[:,:](i8, i8)", parallel=True, cache=True)
def _super_op_from_bcsz_measure_row(dims: int, rank: int):
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(ENGINE.eye(dims, dtype=operator.dtype), operator)
    return operator @ super_op @ operator


@njit("c16[:,:](i8, i8)", parallel=True, cache=True)
def _super_op_from_bcsz_measure_column(dims: int, rank: int):
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(operator, ENGINE.eye(dims, dtype=operator.dtype))
    return operator @ super_op @ operator


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _super_op_from_haar_measure_row(dims: int):
    super_op = _random_unitary_haar(dims)
    super_op = _vectorization_row(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))


@njit("c16[:,:](i8)", parallel=True, cache=True)
def _super_op_from_haar_measure_column(dims: int):
    super_op = _random_unitary_haar(dims)
    super_op = _vectorization_column(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))


# these two can't be jitted (at least globally) because of
# random unitary


def _super_op_from_hermitian_measure_row(dims: int):
    super_op = _random_unitary(dims)
    super_op = _vectorization_row(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))


def _super_op_from_hermitian_measure_column(dims: int):
    super_op = _random_unitary(dims)
    super_op = _vectorization_column(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))


@njit("c16[:,:](c16[:,:,::1], c16[:], i8)", parallel=True, cache=True)
def _kraus_to_stinespring(kraus_ops, initial_state_env, dim_env: int):
    alphas = ENGINE.zeros((dim_env, dim_env, dim_env), dtype=initial_state_env.dtype)
    idx = ENGINE.arange(dim_env)
    for i in prange(dim_env):
        alphas[idx[i], idx[i]] = initial_state_env
    # batched kron product
    dim = kraus_ops.shape[1] * alphas.shape[1]
    prod = ENGINE.zeros((dim, dim), dtype=kraus_ops.dtype)
    for i in prange(len(kraus_ops)):
        prod += ENGINE.kron(kraus_ops[i], alphas[i])
    return prod


@njit("c16[:,:,:](c16[:,::1], c16[:], i8, i8)", parallel=True, cache=True)
def _stinespring_to_kraus(stinespring, initial_state_env, dim: int, dim_env: int):
    stinespring = stinespring.reshape(dim, dim_env, dim, dim_env)
    stinespring = ENGINE.swapaxes(stinespring, 1, 2)
    alphas = ENGINE.eye(dim_env, dtype=stinespring.dtype)
    tmp = ENGINE.empty(stinespring.shape, dtype=stinespring.dtype)
    for i in prange(dim):
        for j in prange(dim):
            tmp[i, j] = alphas @ stinespring[i, j]
    stinespring = tmp.reshape(dim, dim_env, dim + dim_env)
    tmp = ENGINE.empty((2 * dim_env,) + stinespring.shape[:2], dtype=stinespring.dtype)
    tmp[:dim_env] = stinespring[:, :, :dim_env]
    tmp[dim_env:] = stinespring[:, :, dim_env:]
    kraus = ENGINE.empty((tmp.shape[0], initial_state_env.shape[0]), dtype=tmp.dtype)
    for i in prange(tmp.shape[0]):
        kraus[i] = tmp[i] @ initial_state_env
    return kraus.reshape(dim, dim_env, dim_env)


@njit("c16[:,:](c16[:,:])", parallel=True, cache=True)
def _to_choi_row(channel):
    channel = _vectorization_row(channel, channel.shape[-1])
    return ENGINE.outer(channel, ENGINE.conj(channel))


@njit("c16[:,:](c16[:,:])", parallel=True, cache=True)
def _to_choi_column(channel):
    channel = _vectorization_column(channel, channel.shape[-1])
    return ENGINE.outer(channel, ENGINE.conj(channel))


@njit("c16[:,:](c16[:,:])", parallel=True, cache=True)
def _to_liouville_row(channel):
    channel = _to_choi_row(channel)
    return _reshuffling(channel, 1, 2)


@njit("c16[:,:](c16[:,:])", parallel=True, cache=True)
def _to_liouville_column(channel):
    channel = _to_choi_column(channel)
    return _reshuffling(channel, 0, 3)


@njit(
    nbt.Tuple((nbt.complex128[:, :, :], nbt.float64[:]))(
        nbt.float64[:], nbt.complex128[:, :], nbt.float64
    ),
    parallel=True,
    cache=True,
)
def _choi_to_kraus_cp_row(eigenvalues, eigenvectors, precision: float):
    eigv_gt_tol = ENGINE.abs(eigenvalues) > precision
    coefficients = ENGINE.sqrt(eigenvalues[eigv_gt_tol])
    eigenvectors = eigenvectors[eigv_gt_tol]
    dim = int(ENGINE.sqrt(eigenvectors.shape[-1]))
    kraus_ops = coefficients.reshape(-1, 1, 1) * _unvectorization_row(eigenvectors, dim)
    return kraus_ops, coefficients


@njit(
    nbt.Tuple((nbt.complex128[:, :, :], nbt.float64[:]))(
        nbt.float64[:], nbt.complex128[:, :], nbt.float64
    ),
    parallel=True,
    cache=True,
)
def _choi_to_kraus_cp_column(eigenvalues, eigenvectors, precision: float):
    eigv_gt_tol = ENGINE.abs(eigenvalues) > precision
    coefficients = ENGINE.sqrt(eigenvalues[eigv_gt_tol])
    eigenvectors = eigenvectors[eigv_gt_tol]
    dim = int(ENGINE.sqrt(eigenvectors.shape[-1]))
    kraus_ops = coefficients.reshape(-1, 1, 1) * _unvectorization_column(
        eigenvectors, dim
    )
    return kraus_ops, coefficients


@njit("c16[:,:](c16[:,:,:])", parallel=True, cache=True)
def _kraus_to_choi_row(kraus_ops):
    kraus_ops = _vectorization_row(kraus_ops, kraus_ops.shape[-1])
    return kraus_ops.T @ ENGINE.conj(kraus_ops)


@njit("c16[:,:](c16[:,:,:])", parallel=True, cache=True)
def _kraus_to_choi_column(kraus_ops):
    kraus_ops = _vectorization_column(kraus_ops, kraus_ops.shape[-1])
    return kraus_ops.T @ ENGINE.conj(kraus_ops)


class QinfoNumba:
    pass


QINFO = QinfoNumba()


for function in (
    set_seed,
    _pauli_basis,
    _vectorization_row,
    _vectorization_column,
    _unvectorization_row,
    _unvectorization_column,
    _reshuffling,
    _post_sparse_pauli_basis_vectorization,
    _vectorize_pauli_basis_row,
    _vectorize_pauli_basis_column,
    _vectorize_sparse_pauli_basis_row,
    _vectorize_sparse_pauli_basis_column,
    _pauli_to_comp_basis_sparse_row,
    _pauli_to_comp_basis_sparse_column,
    _choi_to_kraus_row,
    _choi_to_kraus_column,
    _random_statevector,
    _random_density_matrix_pure,
    _random_density_matrix_bures,
    _random_density_matrix_hs_ginibre,
    _random_gaussian_matrix,
    _random_hermitian,
    _random_hermitian_semidefinite,
    _random_unitary,
    _random_unitary_haar,
    _sample_from_quantum_mallows_distribution,
    _gamma_delta_matrices,
    _super_op_from_bcsz_measure_row,
    _super_op_from_bcsz_measure_column,
    _super_op_from_haar_measure_row,
    _super_op_from_haar_measure_column,
    _super_op_from_hermitian_measure_row,
    _super_op_from_hermitian_measure_column,
    _kraus_to_stinespring,
    _stinespring_to_kraus,
    _to_choi_row,
    _to_choi_column,
    _to_liouville_row,
    _to_liouville_column,
    _choi_to_kraus_cp_row,
    _choi_to_kraus_cp_column,
    _kraus_to_choi_row,
    _kraus_to_choi_column,
):
    setattr(QINFO, function.__name__, function)


# it would be quite cool and spare us a lot of code repetition if
# we could make a recursive approach like the one below working

SIGNATURES = {
    # "_random_hermitian": ("c16[:,:](i8)", {"parallel": True, "cache": True})
    # "_vectorize_pauli_basis_row": ("c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)", {"parallel": True, "cache": True}),
    # "_vectorize_pauli_basis_column": ("c16[:,::1](i8, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1], f8)", {"parallel": True, "cache": True}),
}


def jit_function(signature, function, **kwargs):
    if isinstance(function, str):
        function = getattr(qinfo, function)
    return njit(signature, **kwargs)(function)


for function, signature in SIGNATURES.items():
    print(function)
    jitted = jit_function(signature[0], function, **signature[1])
    globals()[function] = jitted
    setattr(QINFO, function, jitted)
