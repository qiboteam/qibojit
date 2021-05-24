import psutil
import numpy as np
from numba import prange, njit
NTHREADS = psutil.cpu_count(logical=False)


@njit(parallel=True, cache=True)
def initial_state_vector(nqubits, dtype):
    size = 2 ** nqubits
    state = np.zeros((size,), dtype=dtype)
    state[0] = 1
    return state


@njit(parallel=True, cache=True)
def initial_density_matrix(nqubits, dtype):
    size = 2 ** nqubits
    state = np.zeros((size, size), dtype=dtype)
    state[0, 0] = 1
    return state


def initial_state(nqubits, dtype, is_matrix=False):
    if isinstance(dtype, str):
        dtype = getattr(np, dtype)
    if is_matrix:
        return initial_density_matrix(nqubits, dtype)
    return initial_state_vector(nqubits, dtype)


@njit(cache=True)
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@njit(parallel=True, cache=True)
def collapse_state(state, qubits, result, nqubits, normalize=True):
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


@njit(cache=True, parallel=True)
def measure_frequencies_job(frequencies, probs, thread_nshots, nstates, thread_seed):
    nthreads = len(thread_seed)
    thread_frequencies = np.zeros(shape=(nthreads, frequencies.shape[0]), dtype=frequencies.dtype)
    for n in prange(nthreads):
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


def measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234, nthreads=NTHREADS):
    nstates = 1 << nqubits
    thread_nshots = (nthreads - 1) * [nshots // nthreads]
    thread_nshots.append(thread_nshots[-1] + nshots % nthreads)
    thread_nshots = np.array(thread_nshots)

    np.random.seed(seed)
    thread_seed = np.random.randint(0, int(1e8), size=(nthreads,))

    measure_frequencies_job(frequencies, probs, thread_nshots, nstates, thread_seed)
    return frequencies
