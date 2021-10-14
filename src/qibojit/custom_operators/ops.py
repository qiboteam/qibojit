import numpy as np
from numba import prange, njit


@njit(["complex64[:](complex64[:])",
       "complex128[:](complex128[:])"],
       parallel=True,
       cache=True)
def initial_state_vector(state):
    state[0] = 1
    for i in prange(1, len(state)): # pylint: disable=not-an-iterable
        state[i] = 0
    return state


@njit(["complex64[:,:](complex64[:,:])",
       "complex128[:,:](complex128[:,:])"],
       parallel=True,
       cache=True)
def initial_density_matrix(state):
    for i in prange(len(state)): # pylint: disable=not-an-iterable
        for j in prange(len(state)): # pylint: disable=not-an-iterable
            state[i, j] = 0
    state[0, 0] = 1
    return state


@njit(["int64(int64, int64, UniTuple(int32, 1))",
       "int64(int64, int64, UniTuple(int32, 2))",
       "int64(int64, int64, UniTuple(int32, 3))"],
       cache=True)
def collapse_index(g, h, qubits):
    i = 0
    i += g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@njit(["complex64[:](complex64[:], UniTuple(int32, 1), int64, int64)",
       "complex64[:](complex64[:], UniTuple(int32, 2), int64, int64)",
       "complex64[:](complex64[:], UniTuple(int32, 3), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 1), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 2), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 3), int64, int64)"],
       parallel=True,
       cache=True)
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


@njit(["complex64[:](complex64[:], UniTuple(int32, 1), int64, int64)",
       "complex64[:](complex64[:], UniTuple(int32, 2), int64, int64)",
       "complex64[:](complex64[:], UniTuple(int32, 3), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 1), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 2), int64, int64)",
       "complex128[:](complex128[:], UniTuple(int32, 3), int64, int64)"],
       parallel=True,
       cache=True)
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


@njit(["int64[:](int64[:], float64[:], int64, optional(int64), int64, int64)",
       "int32[:](int32[:], float32[:], int64, optional(int64), int64, int64)",
       "int32[:](int32[:], float64[:], int64, optional(int64), int64, int64)",
       "int64[:](int64[:], float32[:], int64, optional(int64), int64, int64)"],
      cache=True,
      parallel=True)
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

def generate_signature_transpose_state(ndevices=[2,4,8],order=list(range(3,11))):
    list1 = ['a' for a in ndevices]
    list2 = ['a' for a in order]




@njit(["complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 3))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 4))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 5))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 6))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 7))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 8))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 9))",
       "complex64[:](UniTuple(complex64[:], 2), complex64[:], int64, UniTuple(int64, 10))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 3))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 4))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 5))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 6))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 7))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 8))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 9))",
       "complex64[:](UniTuple(complex64[:], 4), complex64[:], int64, UniTuple(int64, 10))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 3))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 4))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 5))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 6))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 7))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 8))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 9))",
       "complex64[:](UniTuple(complex64[:], 8), complex64[:], int64, UniTuple(int64, 10))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 3))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 4))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 5))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 6))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 7))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 8))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 9))",
       "complex128[:](UniTuple(complex128[:], 2), complex128[:], int64, UniTuple(int64, 10))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 3))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 4))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 5))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 6))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 7))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 8))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 9))",
       "complex128[:](UniTuple(complex128[:], 4), complex128[:], int64, UniTuple(int64, 10))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 3))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 4))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 5))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 6))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 7))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 8))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 9))",
       "complex128[:](UniTuple(complex128[:], 8), complex128[:], int64, UniTuple(int64, 10))"],
       cache=True,
       parallel=True)
def transpose_state(pieces, state, nqubits, order):
    nstates = 1 << nqubits
    ndevices = len(pieces)
    npiece = nstates // ndevices
    qubit_exponents = [1 << (nqubits - x - 1) for x in order[::-1]]

    for g in prange(nstates): # pylint: disable=not-an-iterable
        k = 0
        for q in range(nqubits):
            if ((g >> q) % 2):
                k += qubit_exponents[q]
        state[g] = pieces[k // npiece][k % npiece]
    return state


@njit(["void(complex64[:], complex64[:], int64, int64)",
       "void(complex128[:], complex128[:], int64, int64)"],
       cache=True,
       parallel=True)
def swap_pieces(piece0, piece1, new_global, nlocal):
    m = nlocal - new_global - 1
    tk = 1 << m
    nstates = 1 << (nlocal - 1)
    for g in prange(nstates): # pylint: disable=not-an-iterable
        i = ((g >> m) << (m + 1)) + (g & (tk - 1))
        piece0[i + tk], piece1[i] = piece1[i], piece0[i + tk]
