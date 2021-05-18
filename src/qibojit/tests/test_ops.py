import pytest
import itertools
import numpy as np
from qibojit import custom_operators as op
from qibojit.tests.utils import random_state, random_complex, qubits_tensor, ATOL


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("is_matrix", [False, True])
def test_initial_state(dtype, is_matrix):
    final_state =  op.initial_state(4, dtype, is_matrix)
    if is_matrix:
        target_state = np.array([1] + [0]*255, dtype=dtype)
        target_state = np.reshape(target_state, (16, 16))
    else:
        target_state = np.array([1] + [0]*15, dtype=dtype)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits,targets,results",
                         [(2, [0], [1]), (2, [1], [0]), (3, [1], [1]),
                          (4, [1, 3], [1, 0]), (5, [1, 2, 4], [0, 1, 1]),
                          (15, [4, 7], [0, 0]), (16, [8, 12, 15], [1, 0, 1])])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_collapse_state(nqubits, targets, results, dtype):
    atol = 1e-7 if dtype == "complex64" else 1e-14
    state = random_complex((2 ** nqubits,), dtype=dtype)
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = np.reshape(state, nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)

    qubits = sorted(nqubits - np.array(targets, dtype=np.int32) - 1)
    b2d = 2 ** np.arange(len(results) - 1, -1, -1)
    result = int(np.array(results).dot(b2d))
    state = op.collapse_state(state, tuple(qubits), result, nqubits, True)
    np.testing.assert_allclose(state, target_state, atol=atol)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("inttype", ["int32", "int64"])
def test_measure_frequencies(dtype, inttype):
    probs = np.ones(16, dtype=dtype) / 16
    frequencies = np.zeros(16, dtype=inttype)
    frequencies = op.measure_frequencies(frequencies, probs, nshots=1000,
                                         nqubits=4, seed=1234)
    assert np.sum(frequencies) == 1000
    target_frequencies = np.array([64, 85, 66, 53, 47, 55, 66, 65, 57, 48, 72,
                                   65, 60, 68, 72, 57], dtype=inttype)
    np.testing.assert_allclose(frequencies, target_frequencies)


NONZERO = list(itertools.combinations(range(8), r=1))
NONZERO.extend(itertools.combinations(range(8), r=2))
NONZERO.extend(itertools.combinations(range(8), r=3))
NONZERO.extend(itertools.combinations(range(8), r=4))
@pytest.mark.parametrize("nonzero", NONZERO)
def test_measure_frequencies_sparse_probabilities(nonzero):
    probs = np.zeros(8, dtype=np.float64)
    for i in nonzero:
        probs[i] = 1
    probs = probs / np.sum(probs)
    frequencies = np.zeros(8, dtype=np.int64)
    frequencies = op.measure_frequencies(frequencies, probs, nshots=1000,
                                         nqubits=3)
    assert np.sum(frequencies) == 1000
    for i, freq in enumerate(frequencies):
        if i in nonzero:
            assert freq != 0
        else:
            assert freq == 0
