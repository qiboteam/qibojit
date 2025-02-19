import numba.types as nbt
import numpy as np
import qibo.quantum_info.quantum_info as qinfo
from numba import njit, prange
from numba.np.unsafe.ndarray import to_fixed_tuple

ENGINE = qinfo.ENGINE

SIGNATURES = {
    "_vectorization_row": (
        ["c16[:,::1](c16[:,::1], i8)", "c16[:,::1](c16[:,:,::1], i8)"],
        {"parallel": True, "cache": True},
    ),
    "_unvectorization_row": (
        ["c16[:,:,::1](c16[:,::1], i8)", "c16[:,:,::1](c16[:,:,::1], i8)"],
        {"parallel": True, "cache": True},
    ),
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


@njit(
    ["c16[:,::1](c16[:,::1], i8[:])", "c16[:,:,::1](c16[:,:,::1], i8[:])"], cache=True
)
def numba_transpose(array, axes):
    axes = to_fixed_tuple(axes, array.ndim)
    array = np.transpose(array, axes)
    return np.ascontiguousarray(array)


@njit(["c16[:,::1](c16[:,::1], i8)", "c16[:,::1](c16[:,:,::1], i8)"], cache=True)
def _vectorization_column(state, dim):
    indices = ENGINE.arange(state.ndim)
    indices[-2:] = indices[-2:][::-1]
    state = numba_transpose(state, indices)
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

"""
@njit
def _unvectorization_column(state, dim):
    state = ENGINE.reshape(state, (state.shape[0], dim, dim))
    return np.asfortranarray(state)

setattr(QINFO, "_unvectorization_column", _unvectorization_column)
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
def _array_at_2d_indices(array, indices):
    empty = ENGINE.empty(indices[0].shape, dtype=array.dtype)
    for i in prange(len(indices[0])):
        empty[i] = array[indices[0][i], indices[1][i]]
    return empty


@njit(
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[:, ::1]))(
        nbt.complex128[:, ::1], nbt.int64
    ),
    cache=True,
)
def _post_sparse_pauli_basis_vectorization(basis, dim):
    indices = ENGINE.nonzero(basis)
    basis = _array_at_2d_indices(basis, indices)
    basis = basis.reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


setattr(
    QINFO,
    "_post_sparse_pauli_basis_vectorization",
    _post_sparse_pauli_basis_vectorization,
)


for function, signature in SIGNATURES.items():
    print(function)
    jitted = jit_function(signature[0], function, **signature[1])
    globals()[function] = jitted
    setattr(QINFO, function, jitted)


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
    nbt.Tuple((nbt.complex128[:, ::1], nbt.int64[::1]))(
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
    unitary = numba_transpose(unitary, ENGINE.arange(unitary.ndim)[::-1])
    nonzero = ENGINE.nonzero(unitary)
    unitary = _array_at_2d_indices(unitary, nonzero)
    return unitary.reshape(unitary.shape[0], -1), nonzero[1]


setattr(QINFO, "_pauli_to_comp_basis_sparse_row", _pauli_to_comp_basis_sparse_row)
