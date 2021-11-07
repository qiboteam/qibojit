from numba import cuda

@cuda.jit
def initial_density_matrix(state):
    id = cuda.grid(1)
    for index in range(len(state)):
        state[id, index] = 0
    state[0, 0] = 1


@cuda.jit
def initial_state(state):
    id = cuda.grid(1)
    if id == 0:
        state[id] = 1
    else:
        state[id] = 0


@cuda.jit(device=True)
def collapse_index(g, h, qubits):
    i = g
    for j, n in enumerate(qubits):
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + ((h >> j) % 2) * k
    return i


@cuda.jit
def collapse_state_kernel(state, qubits, result, nqubits):
    nsubstates = 1 << len(qubits)

    g = cuda.grid(1)
    for h in range(result):
        state[collapse_index(g, h, qubits)] = 0
    for h in range(result + 1, nsubstates):
        state[collapse_index(g, h, qubits)] = 0

