import pytest
import numpy as np
import qibo
from qibo_sim_numba_cupy import custom_operators as op

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
                         [(5, 4, []), (4, 2, []), (4, 2, []), (3, 0, []),
                          (8, 5, []), (3, 0, [1, 2]), (4, 3, [0, 1, 2]),
                          (5, 3, [1]), (5, 2, [1, 4]), (6, 3, [0, 2, 5]),
                          (6, 3, [0, 2, 4, 5])])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_apply_gate(nqubits, target, controls, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((2, 2), dtype=dtype)

    gate = qibo.gates.Unitary(matrix, target).controlled_by(*controls)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, [target], controls)
    state = op.apply_gate(state, matrix, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target"),
                         [(3, 0), (4, 3), (5, 2), (3, 1)])
@pytest.mark.parametrize("pauli", ["x", "y", "z"])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_apply_pauli_gate(nqubits, target, pauli, dtype):
    qibo.set_backend("numpy")
    state = random_state(nqubits)

    gate = getattr(qibo.gates, pauli.capitalize())(target)
    target_state = gate(np.copy(state))

    qubits = qubits_tensor(nqubits, [target])
    func = getattr(op, "apply_{}".format(pauli))
    state = func(state, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(nqubits, target, controls):
    qibo.set_backend("numpy")
    state = random_state(nqubits)
    theta = 0.1234

    gate = qibo.gates.ZPow(target, theta=theta).controlled_by(*controls)
    target_state = gate(np.copy(state))

    phase = np.exp(1j * theta)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = op.apply_z_pow(state, phase, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state)
