import pytest
import numpy as np
from qibojit import custom_operators as op
from qibojit.tests.utils import random_state, random_complex, qubits_tensor, ATOL


@pytest.mark.parametrize("nqubits,targets,results",
                         [(2, [0], [1]), (2, [1], [0]), (3, [1], [1]),
                          (4, [1, 3], [1, 0]), (5, [1, 2, 4], [0, 1, 1]),
                          (15, [4, 7], [0, 0]), (16, [8, 12, 15], [1, 0, 1])])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_collapse_state(nqubits, targets, results, dtype):
    atol = 1e-7 if dtype == np.complex64 else 1e-14
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
