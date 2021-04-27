from numba import prange, njit


@njit(parallel=True)
def nocontrol_apply(state, gate, kernel, nstates, m):
    tk = 1 << m
    for g in prange(nstates):
        i1 = ((g >> m) << (m + 1)) + (g & (tk - 1))
        i2 = i1 + tk
        state[i1], state[i2] = kernel(state[i1], state[i2], gate)
    return state


@njit(parallel=True)
def multicontrol_apply(state, gate, kernel, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):
        i = 0
        i += g
        for n in qubits:
            k = 1 << n
            i = ((i >> n) << (n + 1)) + (i & (k - 1)) + k
        i1, i2 = i - tk, i
        state[i1], state[i2] = kernel(state[i1], state[i2], gate)
    return state


def apply_gate_base(state, nqubits, target, kernel, qubits=None, gate=None):
    ncontrols = len(qubits) - 1 if qubits is not None else 0
    m = nqubits - target - 1
    nstates = 1 << (nqubits - ncontrols - 1)
    if ncontrols:
        return multicontrol_apply(state, gate, kernel, qubits, nstates, m)
    return nocontrol_apply(state, gate, kernel, nstates, m)


@njit
def apply_gate_kernel(state1, state2, gate):
    return (gate[0, 0] * state1 + gate[0, 1] * state2,
            gate[1, 0] * state1 + gate[1, 1] * state2)

def apply_gate(state, gate, nqubits, target, qubits=None):
    return apply_gate_base(state, nqubits, target, apply_gate_kernel, qubits, gate)


@njit
def apply_x_kernel(state1, state2, gate):
    return state2, state1

def apply_x(state, nqubits, target, qubits=None):
    return apply_gate_base(state, nqubits, target, apply_x_kernel, qubits)


@njit
def apply_y_kernel(state1, state2, gate):
    return -1j * state2, 1j * state1

def apply_y(state, nqubits, target, qubits=None):
    return apply_gate_base(state, nqubits, target, apply_y_kernel, qubits)


@njit
def apply_z_kernel(state1, state2, gate):
    return state1, -state2

def apply_z(state, nqubits, target, qubits=None):
    return apply_gate_base(state, nqubits, target, apply_z_kernel, qubits)


@njit
def apply_z_pow_kernel(state1, state2, gate):
    return state1, gate * state2

def apply_z_pow(state, gate, nqubits, target, qubits=None):
    return apply_gate_base(state, nqubits, target, apply_z_pow_kernel, qubits, gate)
