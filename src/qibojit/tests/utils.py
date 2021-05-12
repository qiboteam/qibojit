import numpy as np


def random_complex(shape, dtype="complex128"):
    x = np.random.random(shape) + 1j * np.random.random(shape)
    return x.astype(dtype)


def random_state(nqubits, dtype="complex128"):
    x = random_complex((2 ** nqubits,), dtype=dtype)
    return x / np.sqrt(np.sum(np.abs(x) ** 2))
