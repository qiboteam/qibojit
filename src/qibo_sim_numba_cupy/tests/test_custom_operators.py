import pytest
import numpy as np
from qibo_sim_numba_cupy import ops


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


@pytest.mark.parametrize(("nqubits", "target", "einsum_str"),
                         [(5, 4, "abcde,Ee->abcdE"),
                          (4, 2, "abcd,Cc->abCd"),
                          (4, 2, "abcd,Cc->abCd"),
                          (3, 0, "abc,Aa->Abc"),
                          (8, 5, "abcdefgh,Ff->abcdeFgh")])
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_apply_gate(nqubits, target, einsum_str, dtype):
    state = random_state(nqubits, dtype=dtype)
    gate = random_complex((2, 2), dtype=dtype)

    target_state = np.reshape(state, nqubits * (2,))
    target_state = np.einsum(einsum_str, target_state, gate)
    target_state = target_state.flatten()

    qubits = qubits_tensor(nqubits, [target])
    state = ops.apply_gate(state, gate, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state)


@pytest.mark.parametrize(("nqubits", "target", "controls", "einsum_str"),
                         [(3, 0, [1, 2], "a,Aa->A"),
                          (4, 3, [0, 1, 2], "a,Aa->A"),
                          (5, 3, [1], "abcd,Cc->abCd"),
                          (5, 2, [1, 4], "abc,Bb->aBc"),
                          (6, 3, [0, 2, 5], "abc,Bb->aBc"),
                          (6, 3, [0, 2, 4, 5], "ab,Bb->aB")])
def test_apply_gate_controlled(nqubits, target, controls, einsum_str):
    state = random_state(nqubits)
    matrix = random_complex((2, 2))

    target_state = np.reshape(state, nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate)
    target_state = target_state.flatten()

    qubits = qubits_tensor(nqubits, [target], controls)
    state = ops.apply_gate(state, matrix, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state)


@pytest.mark.parametrize(("nqubits", "target"),
                         [(3, 0), (4, 3), (5, 2), (3, 1)])
@pytest.mark.parametrize("pauli", ["x", "y", "z"])
def test_apply_pauli_gate(nqubits, target, pauli):
    matrices = {"x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
                "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                "z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}
    state = random_state(nqubits)
    gate = matrices[pauli]

    qubits = qubits_tensor(nqubits, [target])
    target_state = np.copy(state)
    target_state = ops.apply_gate(state, gate, nqubits, target, qubits)

    func = getattr(ops, "apply_{}".format(pauli))
    state = func(state, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state)


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
def test_apply_zpow_gate(nqubits, target, controls):
    import itertools
    phase = np.exp(1j * 0.1234)
    qubits = list(controls)
    qubits.append(target)
    qubits.sort()
    matrix = np.ones(2 ** nqubits, dtype=np.complex128)
    for i, conf in enumerate(itertools.product([0, 1], repeat=nqubits)):
        if np.prod(np.array(conf)[qubits]):
            matrix[i] = phase

    state = random_state(nqubits)
    target_state = np.diag(matrix).dot(state)

    qubits = qubits_tensor(nqubits, [target], controls)
    state = ops.apply_z_pow(state, phase, nqubits, target, qubits)
    np.testing.assert_allclose(state, target_state)
