import itertools

import numpy as np
import pytest
from qibo.quantum_info import random_density_matrix, random_statevector

from qibojit.tests.utils import qubits_tensor, random_complex, set_precision


@pytest.mark.parametrize("is_matrix", [False, True])
def test_zero_state(backend, dtype, is_matrix):
    set_precision(dtype, backend)
    if is_matrix:
        final_state = backend.zero_density_matrix(4)
        target_state = np.array([1] + [0] * 255, dtype=dtype)
        target_state = np.reshape(target_state, (16, 16))
    else:
        final_state = backend.zero_state(4)
        target_state = np.array([1] + [0] * 15, dtype=dtype)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("normalize", [False, True])
def test_identity_density_matrix(backend, dtype, normalize):
    set_precision(dtype, backend)
    norm = 16 if normalize else 1.0
    final_state = backend.identity_density_matrix(4, normalize=normalize)
    target_state = np.eye(16, dtype=dtype) / norm
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("is_matrix", [False, True])
def test_plus_state(backend, dtype, is_matrix):
    set_precision(dtype, backend)
    if is_matrix:
        final_state = backend.plus_density_matrix(4)
        target_state = np.ones((16, 16), dtype=dtype) / 16
    else:
        final_state = backend.plus_state(4)
        target_state = np.ones(16, dtype=dtype) / 4
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(
    "nqubits,targets,results",
    [
        (2, [0], [1]),
        (2, [1], [0]),
        (3, [1], [1]),
        (4, [1, 3], [1, 0]),
        (5, [1, 2, 4], [0, 1, 1]),
        (15, [4, 7], [0, 0]),
        (16, [8, 12, 15], [1, 0, 1]),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_collapse_state(backend, nqubits, targets, results, normalize, dtype):
    atol = 1e-7 if dtype == "complex64" else 1e-14
    state = random_statevector(2**nqubits, backend=backend)
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = np.reshape(np.copy(state), nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    target_state = target_state.flatten()
    if normalize:
        norm = (np.abs(target_state) ** 2).sum()
        target_state = target_state / np.sqrt(norm)

    b2d = 2 ** np.arange(len(results) - 1, -1, -1)
    result = np.array([np.array(results).dot(b2d)])
    state = backend.collapse_state(state, sorted(targets), result, nqubits, normalize)
    backend.assert_allclose(state, target_state, atol=atol)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_collapse_call(backend, density_matrix):
    from qibo import gates
    from qibo.backends import NumpyBackend

    tbackend = NumpyBackend()
    if density_matrix:
        state = random_density_matrix(2**3, backend=tbackend)
    else:
        state = random_statevector(2**3, backend=tbackend)

    gate = gates.M(0, 1, collapse=True)
    np.random.seed(123)
    if density_matrix:
        target_state = gate.apply_density_matrix(tbackend, np.copy(state), 3)
        np.random.seed(123)
        final_state = gate.apply_density_matrix(backend, np.copy(state), 3)
    else:
        target_state = gate.apply(tbackend, np.copy(state), 3)
        np.random.seed(123)
        final_state = gate.apply(backend, np.copy(state), 3)
    backend.assert_allclose(final_state, target_state)


def generate_transpose_qubits(nqubits):
    """Generates global qubits randomly."""
    qubits = np.arange(nqubits)
    np.random.shuffle(qubits)
    return qubits


CONFIG = ((n, generate_transpose_qubits(n)) for _ in range(5) for n in range(3, 11))


@pytest.mark.parametrize("nqubits,qubits", CONFIG)
@pytest.mark.parametrize("ndevices", [2, 4, 8])
def test_transpose_state(backend, nqubits, qubits, ndevices, dtype):
    if backend.platform != "numba":  # pragma: no cover
        pytest.skip(
            f"``transpose_state`` op is not available for {backend.platform} platform."
        )
    qubit_order = list(qubits)
    state = random_statevector(2**nqubits, backend=backend)
    state = backend.cast(state, dtype=dtype)
    state_tensor = np.reshape(state, nqubits * (2,))
    target_state = np.transpose(state_tensor, qubit_order).flatten()
    new_state = np.zeros_like(state)
    state = np.reshape(state, (ndevices, int(state.shape[0]) // ndevices))
    pieces = [state[i] for i in range(ndevices)]
    new_state = backend.ops.transpose_state(pieces, new_state, nqubits, qubit_order)
    backend.assert_allclose(new_state, target_state)


CONFIG = ((n, np.random.randint(1, n)) for _ in range(10) for n in range(4, 11))


@pytest.mark.parametrize("nqubits,local", CONFIG)
def test_swap_pieces_zero_global(backend, nqubits, local, dtype):
    if backend.platform != "numba":  # pragma: no cover
        pytest.skip(
            f"``swap_pieces`` op is not available for {backend.platform} platform."
        )

    from qibo import gates

    state = random_statevector(2**nqubits, backend=backend)
    state = backend.cast(state, dtype=dtype)
    target_state = np.copy(state)
    shape = (2, int(state.shape[0]) // 2)
    state = np.reshape(state, shape)

    set_precision(dtype, backend)
    gate = gates.SWAP(0, local)
    target_state = backend.apply_gate(gate, target_state, nqubits)
    target_state = np.reshape(backend.to_numpy(target_state), shape)
    piece0, piece1 = state[0], state[1]
    backend.ops.swap_pieces(piece0, piece1, local - 1, nqubits - 1)
    backend.assert_allclose(piece0, target_state[0])
    backend.assert_allclose(piece1, target_state[1])


CONFIG = (
    (n, np.random.randint(0, n), np.random.randint(0, n))
    for _ in range(10)
    for n in range(5, 11)
)


@pytest.mark.parametrize("nqubits,qlocal,qglobal", CONFIG)
def test_swap_pieces(backend, nqubits, qlocal, qglobal, dtype):
    if backend.platform != "numba":  # pragma: no cover
        pytest.skip(
            f"``swap_pieces`` op is not available for {backend.platform} platform."
        )

    from qibo import gates

    state = random_statevector(2**nqubits, backend=backend)
    state = backend.cast(state, dtype=dtype)
    target_state = np.copy(state)
    shape = (2, int(state.shape[0]) // 2)

    while qlocal == qglobal:
        qlocal = np.random.randint(0, nqubits)

    transpose_order = (
        [qglobal] + list(range(qglobal)) + list(range(qglobal + 1, nqubits))
    )

    gate = gates.SWAP(qglobal, qlocal)
    target_state = backend.apply_gate(gate, target_state, nqubits)
    target_state = backend.to_numpy(target_state)
    target_state = np.reshape(target_state, nqubits * (2,))
    target_state = np.transpose(target_state, transpose_order)
    target_state = np.reshape(target_state, shape)

    state = np.reshape(state, nqubits * (2,))
    state = np.transpose(state, transpose_order)
    state = np.reshape(state, shape)
    piece0, piece1 = state[0], state[1]
    new_global = qlocal - int(qglobal < qlocal)
    backend.ops.swap_pieces(piece0, piece1, new_global, nqubits - 1)
    backend.assert_allclose(piece0, target_state[0])
    backend.assert_allclose(piece1, target_state[1])


@pytest.mark.parametrize("realtype", ["float32", "float64"])
@pytest.mark.parametrize("inttype", ["int32", "int64"])
@pytest.mark.parametrize("nthreads", [None, 4])
def test_measure_frequencies(backend, realtype, inttype, nthreads):
    probs = np.ones(16, dtype=realtype) / 16
    frequencies = np.zeros(16, dtype=inttype)
    # if backend.name in ["cupy", "cuquantum"]:  # pragma: no cover
    # CI does not test for GPU
    #    with pytest.raises(NotImplementedError):
    #        frequencies = backend.measure_frequencies(frequencies, probs, nshots=1000,
    #                                                     nqubits=4, seed=1234,
    #                                                     nthreads=nthreads)
    # else:
    if nthreads is None:
        nthreads = backend.nthreads
    frequencies = backend.measure_frequencies_op(
        frequencies, probs, nshots=1000, nqubits=4, seed=1234, nthreads=nthreads
    )
    assert np.sum(frequencies) == 1000
    if nthreads == 4:
        target_frequencies = np.array(
            [72, 65, 63, 54, 57, 55, 67, 50, 53, 67, 69, 68, 64, 68, 66, 62],
            dtype=inttype,
        )
        backend.assert_allclose(frequencies, target_frequencies)


NONZERO = list(itertools.combinations(range(8), r=1))
NONZERO.extend(itertools.combinations(range(8), r=2))
NONZERO.extend(itertools.combinations(range(8), r=3))
NONZERO.extend(itertools.combinations(range(8), r=4))
NSHOTS = (len(NONZERO) // 2 + 1) * [1000, 200000]


@pytest.mark.parametrize("nonzero,nshots", zip(NONZERO, NSHOTS))
def test_measure_frequencies_sparse_probabilities(backend, nonzero, nshots):
    probs = np.zeros(8, dtype=np.float64)
    for i in nonzero:
        probs[i] = 1
    probs = probs / np.sum(probs)
    frequencies = backend.sample_frequencies(probs, nshots)
    assert sum(frequencies.values()) == nshots
    for i, freq in frequencies.items():
        assert freq != 0
