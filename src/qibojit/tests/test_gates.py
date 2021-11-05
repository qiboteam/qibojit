import pytest
import numpy as np
import qibo
from qibo import K, gates


ATOL = {"complex64": 1e-5, "complex128": 1e-10}

def qubits_tensor(nqubits, targets, controls=[]):
    qubits = [nqubits - q - 1 for q in targets]
    qubits.extend(nqubits - q - 1 for q in controls)
    return K.cast(sorted(qubits), dtype="int32")


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
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((2, 2), dtype=dtype)

    qibo.set_backend("numpy")
    gate = gates.Unitary(matrix, target).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    qubits = qubits_tensor(nqubits, [target], controls)
    state = K.apply_gate(state, matrix, nqubits, (target,), qubits)
    state = K.to_numpy(state)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (4, 3, []), (5, 2, []), (3, 1, []),
                          (3, 0, [1]), (4, 3, [0, 1]), (5, 2, [1, 3, 4])])
@pytest.mark.parametrize("pauli", ["x", "y", "z"])
def test_apply_pauli_gate(backend, nqubits, target, pauli, controls, dtype):
    state = random_state(nqubits, dtype=dtype)

    qibo.set_backend("numpy")
    gate = getattr(gates, pauli.capitalize())
    gate = gate(target).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    qubits = qubits_tensor(nqubits, [target], controls)
    func = getattr(K, "apply_{}".format(pauli))
    state = func(state, nqubits, (target,), qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(backend, nqubits, target, controls, dtype):
    state = random_state(nqubits, dtype=dtype)
    theta = 0.1234

    qibo.set_backend("numpy")
    gate = gates.U1(target, theta=theta).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    phase = np.exp(1j * theta).astype(dtype)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = K.apply_z_pow(state, phase, nqubits, (target,), qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(5, [3, 4], []), (4, [2, 0], []), (2, [0, 1], []),
                          (8, [6, 3], []), (3, [0, 1], [2]), (4, [1, 3], [0]),
                          (5, [2, 3], [1, 4]), (5, [3, 1], [0, 2]),
                          (6, [2, 5], [0, 1, 3, 4])])
def test_apply_two_qubit_gate(backend, nqubits, targets, controls, dtype):
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((4, 4), dtype=dtype)

    qibo.set_backend("numpy")
    gate = gates.Unitary(matrix, *targets).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.apply_two_qubit_gate(state, matrix, nqubits, targets, qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
def test_apply_swap(backend, nqubits, targets, controls, dtype):
    state = random_state(nqubits, dtype=dtype)

    qibo.set_backend("numpy")
    gate = gates.SWAP(*targets).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.apply_swap(state, nqubits, targets, qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1], []), (4, [2, 0], []), (3, [1, 2], [0]),
                          (4, [0, 1], [2]), (5, [0, 1], [2]), (5, [3, 4], [2]),
                          (4, [0, 3], [1]), (4, [3, 2], [0]), (5, [1, 4], [2]),
                          (6, [1, 3], [0, 4]), (6, [5, 0], [1, 2, 3])])
def test_apply_fsim(backend, nqubits, targets, controls, dtype):
    state = random_state(nqubits, dtype=dtype)
    matrix = random_complex((2, 2), dtype=dtype)
    phi = 0.4321

    qibo.set_backend("numpy")
    gate = gates.GeneralizedfSim(*targets, matrix, phi).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    gate = np.array(list(matrix.flatten()) + [np.exp(-1j * phi)], dtype=dtype)
    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.apply_fsim(state, gate, nqubits, targets, qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1, 2], []), (4, [2, 1, 3], []),
                          (5, [0, 2, 3], []), (8, [2, 6, 3], []),
                          (5, [0, 2, 3, 4], []), (8, [0, 4, 2, 5, 7], []),
                          (7, [0, 2, 4, 3, 6, 5], []), (8, [0, 4, 2, 3, 5, 7, 1], []),
                          (4, [2, 1, 3], [0]), (5, [0, 2, 3], [1]),
                          (8, [2, 6, 3], [4, 7]), (5, [0, 2, 3, 4], [1]),
                          (8, [0, 4, 2, 5, 7], [1, 3]),
                          (10, [0, 4, 2, 5, 9], [1, 3, 7, 8]),
                          (22, [10, 8, 13], []), (22, [11, 20, 13, 4], []),
                          (22, [12, 14, 2, 5, 17], []), (22, [0, 12, 4, 3, 16, 21], []),
                          (22, [0, 14, 20, 13, 5, 17, 21], []),
                          (22, [12, 17, 3], [10]), (22, [21, 6, 13], [14, 7]),
                          (22, [0, 20, 3, 14], [1]),
                          (22, [0, 4, 20, 5, 17], [10, 3]),
                          (22, [10, 20, 4, 3, 16, 5], [12, 19, 15])])
def test_apply_multiqubit_gate(nqubits, targets, controls, dtype):
    state = random_state(nqubits, dtype=dtype)
    rank = 2 ** len(targets)
    matrix = random_complex((rank, rank), dtype=dtype)

    qibo.set_backend("numpy")
    gate = gates.Unitary(matrix, *targets).controlled_by(*controls)
    target_state = gate(K.copy(state))
    qibo.set_backend("qibojit")

    targets = np.array(targets, dtype="int32")
    qubits = qubits_tensor(nqubits, targets, controls) if controls else None
    state = K.apply_multi_qubit_gate(state, matrix, nqubits, targets, qubits)
    K.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize("gatename", ["H", "X", "Z"])
@pytest.mark.parametrize("density_matrix", [False, True])
def test_gates_on_circuit(backend, gatename, density_matrix):
    from qibo.models import Circuit
    if density_matrix:
        state = random_complex((2, 2))
        state = state + np.conj(state.T)
    else:
        state = random_state(1)

    qibo.set_backend("numpy")
    c = Circuit(1, density_matrix=density_matrix)
    c.add(getattr(gates, gatename)(0))
    target_state = c(K.copy(state))

    qibo.set_backend("qibojit")
    c = Circuit(1, density_matrix=density_matrix)
    c.add(getattr(gates, gatename)(0))
    final_state = c(K.copy(state))
    K.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("gatename", ["H", "X", "Z"])
def test_density_matrix_half_calls(backend, gatename):
    state = random_complex((8, 8))
    state = state + np.conj(state.T)
    qibo.set_backend("numpy")
    gate = getattr(gates, gatename)(1)
    gate.nqubits = 3
    gate.density_matrix = True
    if isinstance(gate, gates.MatrixGate):
        target_state = K.density_matrix_half_matrix_call(gate, K.copy(state))
    else:
        target_state = K._density_matrix_half_call(gate, K.copy(state))

    qibo.set_backend("qibojit")
    gate = getattr(gates, gatename)(1)
    gate.nqubits = 3
    gate.density_matrix = True
    if isinstance(gate, gates.MatrixGate):
        final_state = K.density_matrix_half_matrix_call(gate, K.copy(state))
    else:
        final_state = K._density_matrix_half_call(gate, K.copy(state))
    K.assert_allclose(final_state, target_state)
