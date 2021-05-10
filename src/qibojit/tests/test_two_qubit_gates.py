import pytest
import numpy as np
import qibo
from qibojit import custom_operators as op
from qibojit.tests.utils import random_state, random_complex, qubits_tensor, ATOL


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(5, [3, 4], []), (4, [2, 0], []), (2, [0, 1], []),
                          (8, [6, 3], []), (3, [0, 1], [2]), (4, [1, 3], [0]),
                          (5, [2, 3], [1, 4]), (5, [1, 3], [0, 2]),
                          (6, [2, 5], [0, 1, 3, 4])])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_apply_gate(nqubits, targets, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((4, 4), dtype=dtype)

    gate = qibo.gates.Unitary(matrix, *targets).controlled_by(*controls)
    target_state = gate(np.copy(state))

    target1, target2 = targets
    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_two_qubit_gate(state, matrix, nqubits, target1, target2, qubits)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_apply_swap_general(nqubits, targets, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits)

    target1, target2 = targets
    gate = qibo.gates.SWAP(target1, target2).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_swap(state, nqubits, target1, target2, qubits)
    np.testing.assert_allclose(state, target_state)
