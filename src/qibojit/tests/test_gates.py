import pytest
import numpy as np
import qibo
from qibojit import custom_operators as op

ATOL = {"complex64": 1e-6, "complex128": 1e-12}

def qubits_tensor(nqubits, targets, controls=[]):
    qubits = [nqubits - q - 1 for q in targets]
    qubits.extend(nqubits - q - 1 for q in controls)
    return tuple(sorted(qubits))


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return x.astype(dtype)


def random_state(nqubits, dtype="complex128"):
    x = random_complex((2 ** nqubits,), dtype=dtype)
    return x / np.sqrt(np.sum(np.abs(x) ** 2))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(5, 4, []), (4, 2, []), (3, 0, []), (8, 5, []),
                          (3, 0, [1, 2]), (4, 3, [0, 1, 2]),
                          (5, 3, [1]), (5, 2, [1, 4]), (6, 3, [0, 2, 5]),
                          (6, 3, [0, 2, 4, 5])])
def test_apply_gate(backend, nqubits, target, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((2, 2), dtype=dtype)

    gate = qibo.gates.Unitary(matrix, target).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, [target], controls)
    state = op.apply_gate(state, matrix, nqubits, target, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (4, 3, []), (5, 2, []), (3, 1, []),
                          (3, 0, [1]), (4, 3, [0, 1]), (5, 2, [1, 3, 4])])
@pytest.mark.parametrize("pauli", ["x", "y", "z"])
def test_apply_pauli_gate(backend, nqubits, target, pauli, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)

    gate = getattr(qibo.gates, pauli.capitalize())
    gate = gate(target).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, [target], controls)
    func = getattr(op, "apply_{}".format(pauli))
    state = func(state, nqubits, target, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(backend, nqubits, target, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    theta = 0.1234

    gate = qibo.gates.U1(target, theta=theta).controlled_by(*controls)
    target_state = gate(np.copy(state))

    phase = np.exp(1j * theta).astype(dtype)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = op.apply_z_pow(state, phase, nqubits, target, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(5, [3, 4], []), (4, [2, 0], []), (2, [0, 1], []),
                          (8, [6, 3], []), (3, [0, 1], [2]), (4, [1, 3], [0]),
                          (5, [2, 3], [1, 4]), (5, [3, 1], [0, 2]),
                          (6, [2, 5], [0, 1, 3, 4])])
def test_apply_two_qubit_gate(backend, nqubits, targets, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((4, 4), dtype=dtype)

    gate = qibo.gates.Unitary(matrix, *targets).controlled_by(*controls)
    target_state = gate(np.copy(state))

    target1, target2 = targets
    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_two_qubit_gate(state, matrix, nqubits, target1, target2, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
def test_apply_swap(backend, nqubits, targets, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)

    target1, target2 = targets
    gate = qibo.gates.SWAP(target1, target2).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_swap(state, nqubits, target1, target2, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1], []), (4, [2, 0], []), (3, [1, 2], [0]),
                          (4, [0, 1], [2]), (5, [0, 1], [2]), (5, [3, 4], [2]),
                          (4, [0, 3], [1]), (4, [3, 2], [0]), (5, [1, 4], [2]),
                          (6, [1, 3], [0, 4]), (6, [5, 0], [1, 2, 3])])
def test_apply_fsim(backend, nqubits, targets, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((2, 2), dtype=dtype)
    phi = 0.4321

    target1, target2 = targets
    gate = qibo.gates.GeneralizedfSim(target1, target2, matrix, phi).controlled_by(*controls)
    target_state = gate(np.copy(state))

    gate = np.array(list(matrix.flatten()) + [np.exp(-1j * phi)], dtype=dtype)
    qubits = qubits_tensor(nqubits, targets, controls)
    state = op.apply_fsim(state, gate, nqubits, target1, target2, qubits)
    state = op.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))
