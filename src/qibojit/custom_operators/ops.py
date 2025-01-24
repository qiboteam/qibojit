import numpy as np
from numba import njit, prange


@njit(
    ["complex64[:](complex64[:])", "complex128[:](complex128[:])"],
    parallel=True,
    cache=True,
)
def initial_state_vector(state):
    state[0] = 1
    for i in prange(1, len(state)):  # pylint: disable=not-an-iterable
        state[i] = 0
    return state


@njit(
    ["complex64[:,:](complex64[:,:])", "complex128[:,:](complex128[:,:])"],
    parallel=True,
    cache=True,
)
def initial_density_matrix(state):
    for i in prange(len(state)):  # pylint: disable=not-an-iterable
        for j in prange(len(state)):  # pylint: disable=not-an-iterable
            state[i, j] = 0
    state[0, 0] = 1
    return state


@njit("int64(int64, int64, int32[:])", cache=True)
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@njit(
    [
        "complex64[:](complex64[:], int32[:], int64, int64)",
        "complex128[:](complex128[:], int32[:], int64, int64)",
    ],
    parallel=True,
    cache=True,
)
def collapse_state(state, qubits, result, nqubits):
    # qubits = tuple(qubits)
    nstates = 1 << (nqubits - len(qubits))
    nsubstates = 1 << len(qubits)

    for g in prange(nstates):  # pylint: disable=not-an-iterable
        for h in range(result):
            state[collapse_index(g, h, qubits)] = 0
        for h in range(result + 1, nsubstates):
            state[collapse_index(g, h, qubits)] = 0
    return state


@njit(
    [
        "complex64[:](complex64[:], int32[:], int64, int64)",
        "complex128[:](complex128[:], int32[:], int64, int64)",
    ],
    parallel=True,
    cache=True,
)
def collapse_state_normalized(state, qubits, result, nqubits):
    # qubits = tuple(qubits)
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


@njit(
    [
        "int64[:](int64[:], float64[:], int64, optional(int64), int64, int64)",
        "int32[:](int32[:], float32[:], int64, optional(int64), int64, int64)",
        "int32[:](int32[:], float64[:], int64, optional(int64), int64, int64)",
        "int64[:](int64[:], float32[:], int64, optional(int64), int64, int64)",
    ],
    cache=True,
    parallel=True,
)
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
        shot = probs.argmax()
        for i in range(thread_nshots[n]):
            # if i == 0:
            #     # Initial bitstring is the one with the maximum probability
            #     shot = probs.argmax()
            new_shot = (shot + np.random.randint(0, nstates)) % nstates
            # accept or reject move
            if probs[new_shot] / probs[shot] > np.random.random():
                shot = new_shot
            # update frequencies
            frequencies_private[shot] += 1
    frequencies += thread_frequencies.sum(axis=0)
    return frequencies


@njit(cache=True, parallel=True)
def transpose_state(pieces, state, nqubits, order):
    nstates = 1 << nqubits
    ndevices = len(pieces)
    npiece = nstates // ndevices
    qubit_exponents = [1 << (nqubits - x - 1) for x in order[::-1]]

    for g in prange(nstates):  # pylint: disable=not-an-iterable
        k = 0
        for q in range(nqubits):
            if (g >> q) % 2:
                k += qubit_exponents[q]
        state[g] = pieces[k // npiece][k % npiece]
    return state


@njit(
    [
        "void(complex64[:], complex64[:], int64, int64)",
        "void(complex128[:], complex128[:], int64, int64)",
    ],
    cache=True,
    parallel=True,
)
def swap_pieces(piece0, piece1, new_global, nlocal):
    m = nlocal - new_global - 1
    tk = 1 << m
    nstates = 1 << (nlocal - 1)
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m) << (m + 1)) + (g & (tk - 1))
        piece0[i + tk], piece1[i] = piece1[i], piece0[i + tk]
