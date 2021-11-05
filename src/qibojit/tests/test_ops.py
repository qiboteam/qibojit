import pytest
import itertools
import numpy as np
import qibo
from qibo import K
from qibojit.tests.test_gates import qubits_tensor, random_complex, random_state


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("is_matrix", [False, True])
def test_initial_state(backend, precision, is_matrix):
    original_precision = K.precision
    dtype = "complex128" if precision == "double" else "complex64"
    K.set_precision(precision)
    final_state =  K.initial_state(4, is_matrix)
    if is_matrix:
        target_state = np.array([1] + [0]*255, dtype=dtype)
        target_state = np.reshape(target_state, (16, 16))
    else:
        target_state = np.array([1] + [0]*15, dtype=dtype)
    K.assert_allclose(final_state, target_state)
    K.set_precision(original_precision)


@pytest.mark.parametrize("is_matrix", [False, True])
def test_backends_initial_state(backend, dtype, is_matrix):
    final_state =  K.engine.initial_state(4, dtype, is_matrix)
    if is_matrix:
        target_state = np.array([1] + [0]*255, dtype=dtype)
        target_state = np.reshape(target_state, (16, 16))
    else:
        target_state = np.array([1] + [0]*15, dtype=dtype)
    K.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits,targets,results",
                         [(2, [0], [1]), (2, [1], [0]), (3, [1], [1]),
                          (4, [1, 3], [1, 0]), (5, [1, 2, 4], [0, 1, 1]),
                          (15, [4, 7], [0, 0]), (16, [8, 12, 15], [1, 0, 1])])
@pytest.mark.parametrize("normalize", [False, True])
def test_collapse_state(backend, nqubits, targets, results, normalize, dtype):
    atol = 1e-7 if dtype == "complex64" else 1e-14
    state = random_state(nqubits, dtype)
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = np.reshape(state, nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    target_state = target_state.flatten()
    if normalize:
        norm = (np.abs(target_state) ** 2).sum()
        target_state = target_state / np.sqrt(norm)

    qubits = K.cast(sorted(nqubits - np.array(targets, dtype=np.int32) - 1),dtype=np.int32)
    b2d = 2 ** np.arange(len(results) - 1, -1, -1)
    result = int(np.array(results).dot(b2d))
    state = K.collapse_state(state, qubits, result, nqubits, normalize)
    K.assert_allclose(state, target_state, atol=atol)


@pytest.mark.parametrize("gatename", ["H", "X", "Z"])
@pytest.mark.parametrize("density_matrix", [False, True])
def test_collapse_call(backend, gatename, density_matrix):
    from qibo import gates
    if density_matrix:
        state = random_complex((8, 8))
        state = state + np.conj(state.T)
    else:
        state = random_state(3)

    result = [0, 0]
    qibo.set_backend("numpy")
    gate = gates.M(0, 1)
    gate.nqubits = 3
    if density_matrix:
        gate.density_matrix = density_matrix
        target_state = K.density_matrix_collapse(gate, K.copy(state), result)
    else:
        target_state = K.state_vector_collapse(gate, K.copy(state), result)


    qibo.set_backend("qibojit")
    gate = gates.M(0, 1)
    gate.nqubits = 3
    if density_matrix:
        gate.density_matrix = density_matrix
        final_state = K.density_matrix_collapse(gate, K.copy(state), result)
    else:
        final_state = K.state_vector_collapse(gate, K.copy(state), result)
    K.assert_allclose(final_state, target_state)


def generate_transpose_qubits(nqubits):
    """Generates global qubits randomly."""
    qubits = np.arange(nqubits)
    np.random.shuffle(qubits)
    return qubits


CONFIG = ((n, generate_transpose_qubits(n))
          for _ in range(5) for n in range(3, 11))
@pytest.mark.parametrize("nqubits,qubits", CONFIG)
@pytest.mark.parametrize("ndevices", [2, 4, 8])
def test_transpose_state(nqubits, qubits, ndevices, dtype):
    qubit_order = list(qubits)
    state = random_state(nqubits, dtype)
    state_tensor = np.reshape(state, nqubits * (2,))
    target_state = np.transpose(state_tensor, qubit_order).flatten()
    new_state = np.zeros_like(state)
    state = np.reshape(state, (ndevices, int(state.shape[0]) // ndevices))
    pieces = [state[i] for i in range(ndevices)]
    new_state = K.transpose_state(pieces, new_state, nqubits, qubit_order)
    K.assert_allclose(new_state, target_state)


CONFIG = ((n, np.random.randint(1, n)) for _ in range(10) for n in range(4, 11))
@pytest.mark.parametrize("nqubits,local", CONFIG)
def test_swap_pieces_zero_global(nqubits, local, dtype):
    state = random_state(nqubits, dtype)
    target_state = K.copy(state)
    shape = (2, int(state.shape[0]) // 2)
    state = np.reshape(state, shape)

    qubits = qubits_tensor(nqubits, [0, local])
    target_state = K.apply_swap(target_state, nqubits, (0, local), qubits)
    target_state = np.reshape(K.to_numpy(target_state), shape)
    piece0, piece1 = state[0], state[1]
    K.swap_pieces(piece0, piece1, local - 1, nqubits - 1)
    K.assert_allclose(piece0, target_state[0])
    K.assert_allclose(piece1, target_state[1])


CONFIG = ((n, np.random.randint(0, n), np.random.randint(0, n))
          for _ in range(10) for n in range(5, 11))
@pytest.mark.parametrize("nqubits,qlocal,qglobal", CONFIG)
def test_swap_pieces(nqubits, qlocal, qglobal, dtype):
    state = random_state(nqubits, dtype)
    target_state = K.copy(state)
    shape = (2, int(state.shape[0]) // 2)

    while qlocal == qglobal:
        qlocal = np.random.randint(0, nqubits)

    transpose_order = ([qglobal] + list(range(qglobal)) +
                        list(range(qglobal + 1, nqubits)))

    targets = [qglobal, qlocal]
    qubits = qubits_tensor(nqubits, targets)
    target_state = K.apply_swap(target_state, nqubits, targets, qubits)
    target_state = K.to_numpy(target_state)
    target_state = np.reshape(target_state, nqubits * (2,))
    target_state = np.transpose(target_state, transpose_order)
    target_state = np.reshape(target_state, shape)

    state = np.reshape(state, nqubits * (2,))
    state = np.transpose(state, transpose_order)
    state = np.reshape(state, shape)
    piece0, piece1 = state[0], state[1]
    new_global = qlocal - int(qglobal < qlocal)
    K.swap_pieces(piece0, piece1, new_global, nqubits - 1)
    K.assert_allclose(piece0, target_state[0])
    K.assert_allclose(piece1, target_state[1])


@pytest.mark.parametrize("realtype", ["float32", "float64"])
@pytest.mark.parametrize("inttype", ["int32", "int64"])
@pytest.mark.parametrize("nthreads", [None, 4])
def test_measure_frequencies(backend, realtype, inttype, nthreads):
    probs = np.ones(16, dtype=realtype) / 16
    frequencies = np.zeros(16, dtype=inttype)
    if K.engine.name == "cupy":  # pragma: no cover
        # CI does not test for GPU
        with pytest.raises(NotImplementedError):
            frequencies = K.engine.measure_frequencies(frequencies, probs, nshots=1000,
                                                       nqubits=4, seed=1234,
                                                       nthreads=nthreads)
    else:
        frequencies = K.engine.measure_frequencies(frequencies, probs, nshots=1000,
                                                   nqubits=4, seed=1234,
                                                   nthreads=nthreads)
        assert np.sum(frequencies) == 1000
        if nthreads is not None:
            target_frequencies = np.array([72, 65, 63, 54, 57, 55, 67, 50, 53, 67, 69,
                                           68, 64, 68, 66, 62], dtype=inttype)
            K.assert_allclose(frequencies, target_frequencies)


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
    frequencies = K.sample_frequencies(probs, nshots)
    assert np.sum(frequencies) == nshots
    for i, freq in enumerate(frequencies):
        if i in nonzero:
            assert freq != 0
        else:
            assert freq == 0
