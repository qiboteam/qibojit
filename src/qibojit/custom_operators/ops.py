import psutil
import joblib
import numpy as np
from numba import prange, njit
NTHREADS = psutil.cpu_count(logical=False)


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


def initial_state(nqubits, dtype, is_matrix=False):
    if isinstance(dtype, str):
        dtype = getattr(np, dtype)
    if is_matrix:
        return initial_density_matrix(nqubits, dtype)
    return initial_state_vector(nqubits, dtype)


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


@njit
def measure_frequencies_job(frequencies, probs, nshots, nstates, seed):
    frequencies_private = np.zeros_like(frequencies)
    np.random.seed(seed)
    for i in prange(nshots):
        if i == 0:
            # Initial bitstring is the one with the maximum probability
            shot = probs.argmax()
        new_shot = (shot + np.random.randint(0, nstates)) % nstates
        # accept or reject move
        if probs[new_shot] / probs[shot] > np.random.random():
            shot = new_shot
        # update frequencies
        frequencies_private[shot] += 1
    return frequencies_private


def measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234, nthreads=NTHREADS):
    nstates = 1 << nqubits
    thread_nshots = (nthreads - 1) * [nshots // nthreads]
    thread_nshots.append(thread_nshots[-1] + nshots % nthreads)

    np.random.seed(seed)
    thread_seed = np.random.randint(0, int(1e8), size=(nthreads,))

    pool = joblib.Parallel(n_jobs=nthreads, prefer="threads")
    thread_frequencies = pool(
        joblib.delayed(measure_frequencies_job)(frequencies, probs, n, nstates, s)
        for n, s in zip(thread_nshots, thread_seed))
    thread_frequencies = np.array(thread_frequencies)
    frequencies += thread_frequencies.sum(axis=0)
    return frequencies
