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
        gate = gate.control(num_ctrl_qubits=len(controls))
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
