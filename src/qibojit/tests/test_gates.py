import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import PRECISION_TOL
from qibo.models import Circuit
from qibo.quantum_info import (
    random_density_matrix,
    random_statevector,
    random_stochastic_matrix,
    random_unitary,
)

from qibojit.tests.utils import qubits_tensor, random_complex, set_precision

ATOL = {"complex64": 1e-4, "complex128": 1e-10}


@pytest.mark.parametrize(
    ("nqubits", "target", "controls"),
    [
        (5, 4, []),
        (4, 2, []),
        (3, 0, []),
        (8, 5, []),
        (3, 0, [1, 2]),
        (4, 3, [0, 1, 2]),
        (5, 3, [1]),
        (5, 2, [1, 4]),
        (6, 3, [0, 2, 5]),
        (6, 3, [0, 2, 4, 5]),
    ],
)
def test_apply_gate(backend, nqubits, target, controls, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_unitary(2**1, backend=tbackend).astype(dtype)
    gate = gates.Unitary(matrix, target).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "target"), [(4, 1), (6, 5)])
@pytest.mark.parametrize("use_qubits", [False, True])
def test_one_qubit_base(backend, nqubits, target, use_qubits, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_complex((2, 2), dtype=dtype)
    gate = gates.Unitary(matrix, target)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    qubits = qubits_tensor(nqubits, [target]) if use_qubits else None
    state = backend.cast(state)
    matrix = backend.cast(matrix)
    state = backend.one_qubit_base(state, nqubits, target, "apply_gate", matrix, qubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "target", "controls"),
    [
        (3, 0, []),
        (4, 3, []),
        (5, 2, []),
        (3, 1, []),
        (3, 0, [1]),
        (4, 3, [0, 1]),
        (5, 2, [1, 3, 4]),
    ],
)
@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_apply_pauli_gate(backend, nqubits, target, pauli, controls, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = getattr(gates, pauli)
    gate = gate(target).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "target", "controls"),
    [(3, 0, []), (3, 2, [1]), (3, 2, [0, 1]), (6, 1, [0, 2, 4])],
)
def test_apply_zpow_gate(backend, nqubits, target, controls, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    theta = 0.1234
    gate = gates.U1(target, theta=theta).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets", "controls"),
    [
        (5, [3, 4], []),
        (4, [2, 0], []),
        (2, [0, 1], []),
        (8, [6, 3], []),
        (3, [0, 1], [2]),
        (4, [1, 3], [0]),
        (5, [2, 3], [1, 4]),
        (5, [3, 1], [0, 2]),
        (6, [2, 5], [0, 1, 3, 4]),
    ],
)
@pytest.mark.parametrize("density_matrix", [False, True])
def test_apply_two_qubit_gate(
    backend, nqubits, targets, controls, density_matrix, dtype
):
    tbackend = NumpyBackend()
    if density_matrix:
        state = random_density_matrix(2**nqubits, backend=tbackend).astype(dtype)
    else:
        state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_unitary(2**2, backend=tbackend).astype(dtype)
    gate = gates.Unitary(matrix, *targets).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    if density_matrix:
        target_state = tbackend.apply_gate_density_matrix(gate, np.copy(state), nqubits)
        state = backend.apply_gate_density_matrix(gate, np.copy(state), nqubits)
    else:
        target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
        state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(("nqubits", "targets"), [(5, [3, 4]), (4, [2, 0])])
@pytest.mark.parametrize("use_qubits", [False, True])
def test_apply_two_qubit_base(backend, nqubits, targets, use_qubits, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_complex((4, 4), dtype=dtype)
    gate = gates.Unitary(matrix, *targets)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    qubits = qubits_tensor(nqubits, targets) if use_qubits else None
    state = backend.cast(state)
    matrix = backend.cast(matrix)
    state = backend.two_qubit_base(
        state, nqubits, targets[0], targets[1], "apply_two_qubit_gate", matrix, qubits
    )
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets"),
    [
        (2, [0, 1]),
        (3, [0, 2]),
        (4, [1, 3]),
        (3, [1, 2]),
        (4, [0, 2]),
        (4, [2, 3]),
        (5, [3, 4]),
        (6, [1, 4]),
    ],
)
def test_apply_cy(backend, nqubits, targets, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = gates.CY(*targets)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets"),
    [
        (2, [0, 1]),
        (3, [0, 2]),
        (4, [1, 3]),
        (3, [1, 2]),
        (4, [0, 2]),
        (4, [2, 3]),
        (5, [3, 4]),
        (6, [1, 4]),
    ],
)
def test_apply_csx(backend, nqubits, targets, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = gates.CSX(*targets)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets"),
    [
        (2, [0, 1]),
        (3, [0, 2]),
        (4, [1, 3]),
        (3, [1, 2]),
        (4, [0, 2]),
        (4, [2, 3]),
        (5, [3, 4]),
        (6, [1, 4]),
    ],
)
def test_apply_csxdg(backend, nqubits, targets, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = gates.CSXDG(*targets)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets"),
    [
        (3, [0, 1, 2]),
        (4, [1, 2, 3]),
        (4, [0, 2, 3]),
        (4, [2, 3, 1]),
        (5, [3, 4, 2]),
        (6, [1, 2, 4]),
    ],
)
def test_apply_deutsch(backend, nqubits, targets, dtype):
    theta = np.random.rand()
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = gates.DEUTSCH(*targets, theta)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets", "controls"),
    [
        (2, [0, 1], []),
        (3, [0, 2], []),
        (4, [1, 3], []),
        (3, [1, 2], [0]),
        (4, [0, 2], [1]),
        (4, [2, 3], [0]),
        (5, [3, 4], [1, 2]),
        (6, [1, 4], [0, 2, 5]),
    ],
)
def test_apply_swap(backend, nqubits, targets, controls, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    gate = gates.SWAP(*targets).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets", "controls"),
    [
        (3, [0, 1], []),
        (4, [2, 0], []),
        (3, [1, 2], [0]),
        (4, [0, 1], [2]),
        (5, [0, 1], [2]),
        (5, [3, 4], [2]),
        (4, [0, 3], [1]),
        (4, [3, 2], [0]),
        (5, [1, 4], [2]),
        (6, [1, 3], [0, 4]),
        (6, [5, 0], [1, 2, 3]),
    ],
)
def test_apply_fsim(backend, nqubits, targets, controls, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_complex((2, 2), dtype=dtype)
    phi = 0.4321
    gate = gates.GeneralizedfSim(*targets, matrix, phi).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets", "controls"),
    [
        (3, [0, 1, 2], []),
        (4, [2, 1, 3], []),
        (5, [0, 2, 3], []),
        (8, [2, 6, 3], []),
        (5, [0, 2, 3, 4], []),
        (8, [0, 4, 2, 5, 7], []),
        (7, [0, 2, 4, 3, 6, 5], []),
        (8, [0, 4, 2, 3, 5, 7, 1], []),
        (4, [2, 1, 3], [0]),
        (5, [0, 2, 3], [1]),
        (8, [2, 6, 3], [4, 7]),
        (5, [0, 2, 3, 4], [1]),
        (8, [0, 4, 2, 5, 7], [1, 3]),
    ],
)
@pytest.mark.parametrize("density_matrix", [False, True])
def test_apply_multiqubit_gate(
    backend, nqubits, targets, controls, density_matrix, dtype
):
    tbackend = NumpyBackend()
    if density_matrix:
        state = random_density_matrix(2**nqubits, backend=tbackend).astype(dtype)
    else:
        state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    rank = 2 ** len(targets)
    matrix = random_complex((rank, rank), dtype=dtype)
    gate = gates.Unitary(matrix, *targets).controlled_by(*controls)

    set_precision(dtype, backend, tbackend)
    if density_matrix:
        target_state = tbackend.apply_gate_density_matrix(gate, np.copy(state), nqubits)
        state = backend.apply_gate_density_matrix(gate, np.copy(state), nqubits)
    else:
        target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
        state = backend.apply_gate(gate, np.copy(state), nqubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize(
    ("nqubits", "targets", "controls"),
    [
        (10, [0, 4, 2, 5, 9], [1, 3, 7, 8]),
        (22, [10, 8, 13], []),
        (22, [11, 20, 13, 4], []),
        (22, [12, 14, 2, 5, 17], []),
        (22, [0, 12, 4, 3, 16, 21], []),
        (22, [0, 14, 20, 13, 5, 17, 21], []),
        (22, [12, 17, 3], [10]),
        (22, [21, 6, 13], [14, 7]),
        (22, [0, 20, 3, 14], [1]),
        (22, [0, 4, 20, 5, 17], [10, 3]),
        (22, [10, 20, 4, 3, 16, 5], [12, 19, 15]),
    ],
)
def test_apply_multiqubit_gate_large(backend, nqubits, targets, controls, dtype):
    test_apply_multiqubit_gate(backend, nqubits, targets, controls, False, dtype)


@pytest.mark.parametrize(("nqubits", "targets"), [(5, [2, 3, 4]), (4, [2, 0, 1])])
@pytest.mark.parametrize("use_qubits", [False, True])
def test_apply_multi_qubit_base(backend, nqubits, targets, use_qubits, dtype):
    tbackend = NumpyBackend()
    state = random_statevector(2**nqubits, backend=tbackend).astype(dtype)
    matrix = random_complex((8, 8), dtype=dtype)
    gate = gates.Unitary(matrix, *targets)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_gate(gate, np.copy(state), nqubits)
    if use_qubits:
        qubits = backend.cast(qubits_tensor(nqubits, targets), dtype="int32")
    else:
        qubits = None
    state = backend.cast(state)
    matrix = backend.cast(matrix)
    state = backend.multi_qubit_base(state, nqubits, targets, matrix, qubits)
    backend.assert_allclose(state, target_state, atol=ATOL.get(dtype))


@pytest.mark.parametrize("gatename", ["H", "X", "Y", "Z"])
@pytest.mark.parametrize("density_matrix", [False, True])
def test_gates_on_circuit(backend, gatename, density_matrix):
    tbackend = NumpyBackend()
    if density_matrix:
        state = random_density_matrix(2**1, backend=tbackend)
    else:
        state = random_statevector(2**1, backend=tbackend)

    c = Circuit(1, density_matrix=density_matrix)
    c.add(getattr(gates, gatename)(0))

    target_state = tbackend.execute_circuit(c, np.copy(state))
    final_state = backend.execute_circuit(c, np.copy(state))
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(
    "gatename,params",
    [
        ("CRX", {"theta": 0.1}),
        ("CRY", {"theta": 0.1}),
        ("CRZ", {"theta": 0.1}),
        ("CU1", {"theta": 0.1}),
        ("CU2", {"phi": 0.1, "lam": 0.2}),
        ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
        ("fSim", {"theta": 0.1, "phi": 0.2}),
    ],
)
@pytest.mark.parametrize("density_matrix", [False, True])
def test_parametrized_gates_on_circuit(backend, gatename, params, density_matrix):
    from qibo.models import Circuit

    tbackend = NumpyBackend()
    if density_matrix:
        state = random_density_matrix(2**2, backend=tbackend)
    else:
        state = random_statevector(2**2, backend=tbackend)

    c = Circuit(2, density_matrix=density_matrix)
    c.add(getattr(gates, gatename)(0, 1, **params))

    target_state = tbackend.execute_circuit(c, np.copy(state))
    final_state = backend.execute_circuit(c, np.copy(state))
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("gatename", ["H", "X", "Z"])
def test_density_matrix_half_calls(backend, gatename):
    tbackend = NumpyBackend()
    state = random_density_matrix(2**3, backend=tbackend)
    gate = getattr(gates, gatename)(1)

    target_state = tbackend.apply_gate_half_density_matrix(gate, np.copy(state), 3)
    final_state = backend.apply_gate_half_density_matrix(gate, np.copy(state), 3)
    backend.assert_allclose(final_state, target_state)


def test_unitary_channel(backend, dtype):
    tbackend = NumpyBackend()
    a1 = gates.X(0).matrix(tbackend)
    a2 = gates.CNOT(0, 1).matrix(tbackend)
    qubits = [(0,), (2, 3)]
    probs = [0.4, 0.3]
    matrices = list(zip(probs, [a1, a2]))
    channel = gates.UnitaryChannel(qubits, matrices)
    state = random_density_matrix(2**4, backend=tbackend).astype(dtype)

    set_precision(dtype, backend, tbackend)
    target_state = tbackend.apply_channel_density_matrix(channel, np.copy(state), 4)
    final_state = backend.apply_channel_density_matrix(channel, np.copy(state), 4)
    backend.assert_allclose(final_state, target_state)


def test_readout_error_channel(backend):
    nqubits = 1
    d = 2**nqubits

    rho = random_density_matrix(d, seed=1, backend=backend)
    P = random_stochastic_matrix(d, seed=1, backend=backend)

    probability_sum = gates.ReadoutErrorChannel(0, P).apply_density_matrix(
        backend, rho, 1
    )
    probability_sum = np.diag(probability_sum).sum().real

    backend.assert_allclose(probability_sum - 1 < PRECISION_TOL, True)
