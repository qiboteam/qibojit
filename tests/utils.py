import numpy as np


def qubits_tensor(nqubits, targets, controls=[]):
    qubits = [nqubits - q - 1 for q in targets]
    qubits.extend(nqubits - q - 1 for q in controls)
    return np.array(sorted(qubits), dtype="int32")


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return x.astype(dtype)


def set_dtype(dtype, *backends):
    for backend in backends:
        backend.set_dtype(dtype)
