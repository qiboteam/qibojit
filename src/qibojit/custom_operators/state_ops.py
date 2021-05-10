import numpy as np
from numba import prange, njit


@njit
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = (i >> n) << (n + 1) + (i & (k - 1))
        i += ((h >> j) % 2) * k
    return i


@njit(parallel=True)
def collapse_state(state, qubits, result, nqubits, normalize):
    nstates = 1 << (nqubits - len(qubits))
    nsubstates = 1 << len(qubits)

    norms = 0
    for g in prange(nstates):
        for h in range(result):
            state[collapse_index(g, h, qubits)] = 0
        norms += np.abs(state[collapse_index(g, result, qubits)]) ** 2
        for h in range(result + 1, nsubstates):
            state[collapse_index(g, h, qubits)] = 0

    if normalize:
        norm = np.sqrt(norms)
        for g in prange(nstates):
            i = collapse_index(g, result, qubits)
            state[i] = state[i] / norm

    return state
