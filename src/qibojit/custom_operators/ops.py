import numpy as np
from numba import prange, njit


@njit(parallel=True, cache=True)
def initial_state_vector(state):
    state[0] = 1
    for i in prange(1, len(state)): # pylint: disable=not-an-iterable
        state[i] = 0
    return state


@njit(parallel=True, cache=True)
def initial_density_matrix(state):
    for i in prange(len(state)): # pylint: disable=not-an-iterable
        for j in prange(len(state)): # pylint: disable=not-an-iterable
            state[i, j] = 0
    state[0, 0] = 1
    return state


@njit(cache=True)
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@njit(parallel=True, cache=True)
def collapse_state(state, qubits, result, nqubits):
    qubits = tuple(qubits)
    nstates = 1 << (nqubits - len(qubits))
    nsubstates = 1 << len(qubits)

    for g in prange(nstates):  # pylint: disable=not-an-iterable
        for h in range(result):
            state[collapse_index(g, h, qubits)] = 0
        for h in range(result + 1, nsubstates):
            state[collapse_index(g, h, qubits)] = 0
    return state


@njit(parallel=True, cache=True)
def collapse_state_normalized(state, qubits, result, nqubits):
    qubits = tuple(qubits)
    nstates = 1 << (nqubits - len(qubits))
    nsubstates = 1 << len(qubits)

    norms = 0
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        for h in range(result):
            state[collapse_index(g, h, qubits)] = 0
        norms += np.abs(state[collapse_index(g, result, qubits)]) ** 2
        for h in range(result + 1, nsubstates):
            state[collapse_index(g, h, qubits)] = 0

    norm = np.sqrt(norms)
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = collapse_index(g, result, qubits)
        state[i] = state[i] / norm
    return state


@njit(cache=True, parallel=True)
def measure_frequencies(frequencies, probs, nshots, nqubits, seed, nthreads):
    nstates = frequencies.shape[0]
    thread_nshots = np.zeros(nthreads, dtype=frequencies.dtype)
    thread_nshots[:] = nshots // nthreads
    thread_nshots[-1] += nshots % nthreads

    np.random.seed(seed)
    thread_seed = [np.random.randint(0, int(1e8)) for _ in range(nthreads)]

    thread_frequencies = np.zeros(shape=(nthreads, nstates), dtype=frequencies.dtype)
    for n in prange(nthreads):  # pylint: disable=not-an-iterable
        frequencies_private = thread_frequencies[n]
        np.random.seed(thread_seed[n])
        for i in range(thread_nshots[n]):
            if i == 0:
                # Initial bitstring is the one with the maximum probability
                shot = probs.argmax()
            new_shot = (shot + np.random.randint(0, nstates)) % nstates
            # accept or reject move
            if probs[new_shot] / probs[shot] > np.random.random():
                shot = new_shot
            # update frequencies
            frequencies_private[shot] += 1
    frequencies += thread_frequencies.sum(axis=0)
    return frequencies
