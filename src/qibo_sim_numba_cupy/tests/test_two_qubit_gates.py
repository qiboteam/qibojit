import pytest
import numpy as np
import qibo
from qibo_sim_numba_cupy import custom_operators as op
from qibo_sim_numba_cupy.tests.utils import random_state, qubits_tensor, ATOL


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
