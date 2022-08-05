# -*- coding: utf-8 -*-
import numpy as np


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = [nqubits - q - 1 for q in targets]
    qubits.extend(nqubits - q - 1 for q in controls)
    return np.array(sorted(qubits), dtype="int32")


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return x.astype(dtype)


def random_state(nqubits, dtype="complex128"):
    x = random_complex((2**nqubits,), dtype=dtype)
    return x / np.sqrt(np.sum(np.abs(x) ** 2))


def random_density_matrix(nqubits, dtype="complex128"):
    x = random_complex(2 * (2**nqubits,), dtype=dtype)
    return x / np.trace(x)


def random_unitary(nqubits, dtype="complex128"):
    from scipy.linalg import expm

    shape = 2 * (2**nqubits,)
    m = random_complex(shape, dtype=dtype)
    return expm(1j * (m + m.T.conj()))


def set_precision(dtype, *backends):
    if dtype == "complex64":
        precision = "single"
    else:
        precision = "double"
    for backend in backends:
        backend.set_precision(precision)
