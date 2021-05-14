from numba import prange, njit


@njit(parallel=True)
def nocontrol_apply(state, gate, kernel, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):
        i = ((g >> m1) << (m1 + 1)) + (g & (tk1 - 1))
        i = ((i >> m2) << (m2 + 1)) + (i & (tk2 - 1))
        i1, i2 = i + uk1, i + uk2
        i3 = i + tk1 + tk2
        state[i], state[i1], state[i2], state[i3] = kernel(
            state[i], state[i1], state[i2], state[i3], gate)
    return state


@njit(parallel=True)
def multicontrol_apply(state, gate, kernel, qubits, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):
        i = 0
        i += g
        for m in qubits:
            k = 1 << m
            i = ((i >> m) << (m + 1)) + (i & (k - 1)) + k
        i1, i2 = i - uk2, i - uk1
        i0 = i1 - uk1
        state[i0], state[i1], state[i2], state[i] = kernel(
            state[i0], state[i1], state[i2], state[i], gate)
    return state


def apply_gate_base(state, nqubits, target1, target2, kernel, qubits=None, gate=None):
    ncontrols = len(qubits) - 2 if qubits is not None else 0
    if target1 > target2:
        swap_targets = True
        m1 = nqubits - target1 - 1
        m2 = nqubits - target2 - 1
    else:
        swap_targets = False
        m1 = nqubits - target2 - 1
        m2 = nqubits - target1 - 1
    nstates = 1 << (nqubits - 2 - ncontrols)
    if ncontrols:
        return multicontrol_apply(state, gate, kernel, qubits, nstates, m1, m2, swap_targets)
    return nocontrol_apply(state, gate, kernel, nstates, m1, m2, swap_targets)


@njit
def apply_two_qubit_gate_kernel(state0, state1, state2, state3, gate):
    buffer0, buffer1, buffer2 = state0, state1, state2
    state0 = (gate[0, 0] * state0 + gate[0, 1] * state1 +
              gate[0, 2] * state2 + gate[0, 3] * state3)
    state1 = (gate[1, 0] * buffer0 + gate[1, 1] * state1 +
              gate[1, 2] * state2 + gate[1, 3] * state3)
    state2 = (gate[2, 0] * buffer0 + gate[2, 1] * buffer1 +
              gate[2, 2] * state2 + gate[2, 3] * state3)
    state3 = (gate[3, 0] * buffer0 + gate[3, 1] * buffer1 +
              gate[3, 2] * buffer2 + gate[3, 3] * state3)
    return state0, state1, state2, state3

def apply_two_qubit_gate(state, gate, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2,
                           apply_two_qubit_gate_kernel,
                           qubits, gate)


@njit
def apply_swap_kernel(state0, state1, state2, state3, gate):
    return state0, state2, state1, state3

def apply_swap(state, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2, apply_swap_kernel,
                           qubits)


@njit
def apply_fsim_kernel(state0, state1, state2, state3, gate):
    state1, state2 = (gate[0] * state1 + gate[1] * state2,
                      gate[2] * state1 + gate[3] * state2)
    state3 *= gate[4]
    return state0, state1, state2, state3

def apply_fsim(state, gate, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2, apply_fsim_kernel,
                           qubits, gate)
