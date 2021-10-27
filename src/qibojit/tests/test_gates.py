import pytest
import numpy as np
import qiskit
from scipy.linalg import expm
from qiskit.providers.aer import StatevectorSimulator
from qibojit.tests import backend as K

ATOL = {"complex64": 1e-6, "complex128": 1e-13}


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = [nqubits - q - 1 for q in targets]
    qubits.extend(nqubits - q - 1 for q in controls)
    return K.cast(sorted(qubits), dtype="int32")


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return x.astype(dtype)


def random_unitary(shape, dtype="complex128"):
    x = random_complex(shape, dtype)
    return expm(1j * (x + np.conj(x.T)))


def random_state(nqubits, dtype="complex128"):
    x = random_complex((2 ** nqubits,), dtype=dtype)
    return x / np.sqrt(np.sum(np.abs(x) ** 2))


def execute_qiskit(gates, targets, nqubits, state):
    simulator = StatevectorSimulator()
    circuit = qiskit.QuantumCircuit(nqubits)
    state = np.reshape(np.copy(state), nqubits * (2,))
    state = np.transpose(state, range(nqubits - 1, -1, -1))
    state = np.reshape(state, (2 ** nqubits,))
    circuit.initialize(state)
    for gate, targets in zip(gates, targets):
        circuit.append(gate, targets)
    try:
        result = simulator.run(circuit).result()
        state = result.get_statevector(circuit)
    except qiskit.exceptions.QiskitError:
        circuit = qiskit.transpile(circuit, backend=simulator)
        result = simulator.run(circuit).result()
        state = result.get_statevector(circuit)
    state = np.reshape(state, nqubits * (2,))
    state = np.transpose(state, range(nqubits - 1, -1, -1))
    return np.reshape(state, (2 ** nqubits,))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(5, 4, []), (4, 2, []), (3, 0, []), (8, 5, []),
                          (3, 0, [1, 2]), (4, 3, [0, 1, 2]),
                          (5, 3, [1]), (5, 2, [1, 4]), (6, 3, [0, 2, 5]),
                          (6, 3, [0, 2, 4, 5])])
def test_apply_gate(nqubits, target, controls, dtype):
    state = random_state(nqubits)
    matrix = random_unitary((2, 2))

    gate = qiskit.extensions.UnitaryGate(matrix)
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + [target]], nqubits, state)

    state = state.astype(dtype)
    matrix = matrix.astype(dtype)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = K.one_qubit_base(state, nqubits, target, "apply_gate", qubits, matrix)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (4, 3, []), (5, 2, []), (3, 1, []),
                          (3, 0, [1]), (4, 3, [0, 1]), (5, 2, [1, 3, 4])])
@pytest.mark.parametrize("pauli", ["x", "y", "z"])
def test_apply_pauli_gate(nqubits, target, pauli, controls, dtype):
    state = random_state(nqubits)

    circuit = qiskit.QuantumCircuit(1)
    getattr(circuit, pauli)(0)
    gate = circuit.to_gate()
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + [target]], nqubits, state)

    state = state.astype(dtype)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = K.one_qubit_base(state, nqubits, target, f"apply_{pauli}", qubits)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(nqubits, target, controls, dtype):
    state = random_state(nqubits)
    theta = 0.1234

    circuit = qiskit.QuantumCircuit(1)
    circuit.u1(theta, 0)
    gate = circuit.to_gate()
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + [target]], nqubits, state)

    state = state.astype(dtype)
    phase = np.exp(1j * theta).astype(dtype)
    qubits = qubits_tensor(nqubits, [target], controls)
    state = K.one_qubit_base(state, nqubits, target, f"apply_z_pow", qubits, phase)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(5, [3, 4], []), (4, [2, 0], []), (2, [0, 1], []),
                          (8, [6, 3], []), (3, [0, 1], [2]), (4, [1, 3], [0]),
                          (5, [2, 3], [1, 4]), (5, [3, 1], [0, 2]),
                          (6, [2, 5], [0, 1, 3, 4])])
def test_apply_two_qubit_gate(nqubits, targets, controls, dtype):
    state = random_state(nqubits)
    matrix = random_unitary((4, 4))

    gate = qiskit.extensions.UnitaryGate(matrix)
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + targets[::-1]], nqubits, state)

    state = state.astype(dtype)
    matrix = matrix.astype(dtype)
    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.two_qubit_base(state, nqubits, *targets, f"apply_two_qubit_gate", qubits, matrix)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
def test_apply_swap(nqubits, targets, controls, dtype):
    state = random_state(nqubits)

    circuit = qiskit.QuantumCircuit(2)
    circuit.swap(0, 1)
    gate = circuit.to_gate()
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + targets], nqubits, state)

    state = state.astype(dtype)
    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.two_qubit_base(state, nqubits, *targets, f"apply_swap", qubits)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1], []), (4, [2, 0], []), (3, [1, 2], [0]),
                          (4, [0, 1], [2]), (5, [0, 1], [2]), (5, [3, 4], [2]),
                          (4, [0, 3], [1]), (4, [3, 2], [0]), (5, [1, 4], [2]),
                          (6, [1, 3], [0, 4]), (6, [5, 0], [1, 2, 3])])
def test_apply_fsim(nqubits, targets, controls, dtype):
    state = random_state(nqubits)
    matrix = random_unitary((2, 2))
    phi = 0.4321

    qmatrix = np.zeros((4, 4), dtype=np.complex128)
    qmatrix[0, 0] = 1
    qmatrix[1:3, 1:3] = matrix
    qmatrix[3, 3] = np.exp(-1j * phi)
    gate = qiskit.extensions.UnitaryGate(qmatrix)
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + targets[::-1]], nqubits, state)

    state = state.astype(dtype)
    matrix = list(matrix.ravel())
    matrix.append(np.exp(-1j *phi))
    matrix = np.array(matrix, dtype=dtype)
    qubits = qubits_tensor(nqubits, targets, controls)
    state = K.two_qubit_base(state, nqubits, *targets, f"apply_fsim", qubits, matrix)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(3, [0, 1, 2], []), (4, [2, 1, 3], []),
                          (5, [0, 2, 3], []), (8, [2, 6, 3], []),
                          (5, [0, 2, 3, 4], []), (8, [0, 4, 2, 5, 7], []),
                          (7, [0, 2, 4, 3, 6, 5], []), (8, [0, 4, 2, 3, 5, 7, 1], []),
                          (4, [2, 1, 3], [0]), (5, [0, 2, 3], [1]),
                          (8, [2, 6, 3], [4, 7]), (5, [0, 2, 3, 4], [1]),
                          (8, [0, 4, 2, 5, 7], [1, 3]),
                          (10, [0, 4, 2, 5, 9], [1, 3, 7, 8])])
def test_apply_multi_qubit_gate(nqubits, targets, controls, dtype):
    state = random_state(nqubits)
    rank = 2 ** len(targets)
    matrix = random_unitary((rank, rank))

    gate = qiskit.extensions.UnitaryGate(matrix)
    if controls:
        gate = gate.control(len(controls))
    target_state = execute_qiskit([gate], [controls + targets[::-1]], nqubits, state)

    state = state.astype(dtype)
    matrix = matrix.astype(dtype)
    targets = np.array(targets, dtype="int32")
    qubits = qubits_tensor(nqubits, targets, controls) if controls else None
    state = K.multi_qubit_base(state, nqubits, targets, qubits, matrix)
    state = K.to_numpy(state)
    np.testing.assert_allclose(state, target_state, atol=ATOL.get(dtype))
