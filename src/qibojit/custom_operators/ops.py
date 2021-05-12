import numpy as np
from numba import prange, njit


@njit(parallel=True)
def initial_state_vector(nqubits, dtype):
    size = 2 ** nqubits
    state = np.zeros((size,), dtype=dtype)
    state[0] = 1
    return state


@njit(parallel=True)
def initial_density_matrix(nqubits, dtype):
    size = 2 ** nqubits
    state = np.zeros((size, size), dtype=dtype)
    state[0, 0] = 1
    return state


@njit
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@njit(parallel=True)
def collapse_state(state, qubits, result, nqubits, normalize):
    qubits = tuple(qubits)
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


@njit(parallel=True)
def measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234):
    nstates = 1 << nqubits
    np.random.seed(seed)
    # FIXME: sum(frequencies) == nshots does not work probably due to
    # parallelization
    # Initial bitstring is the one with the maximum probability
    for i in prange(nshots):
        if i == 0:
            shot = probs.argmax()
        new_shot = (shot + np.random.randint(0, nstates)) % nstates
        # accept or reject move
        if probs[new_shot] / probs[shot] > np.random.random():
            shot = new_shot
        # update frequencies
        frequencies[shot] += 1
    return frequencies
