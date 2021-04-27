from numba import prange, njit


@njit(parallel=True)
def nocontrol_apply(state, gate, kernel, nstates, m1, m2):
    tk1, tk2 = 1 << m1, 1 << m2
    for g in prange(nstates):
        i = ((g >> m1) << (m1 + 1)) + (g & (tk1 - 1))
        i = ((i >> m2) << (m2 + 1)) + (i & (tk2 - 1))
        i1, i2 = i + tk2, i + tk1
        state[i1], state[i2] = kernel(state[i1], state[i2])
    return state


@njit(parallel=True)
def multicontrol_apply(state, gate, kernel, qubits, nstates, m1, m2):
    tk1, tk2 = 1 << m1, 1 << m2
    ncontrols = len(qubits)
    for g in prange(nstates):
        i = 0
        i += g
        for m in qubits:
            k = 1 << m
            i = ((i >> m) << (m + 1)) + (i & (k - 1)) + k
        i1, i2 = i - tk1, i - tk2
        state[i1], state[i2] = kernel(state[i1], state[i2])
    return state


def apply_gate_base(state, nqubits, target1, target2, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 2 if qubits is not None else 0
        t1, t2 = max(target1, target2), min(target1, target2)
        m1, m2 = nqubits - t1 - 1, nqubits - t2 - 1
        nstates = 1 << (nqubits - 2 - ncontrols)
        if ncontrols:
            return multicontrol_apply(state, gate, kernel, qubits, nstates, m1, m2)
        return nocontrol_apply(state, gate, kernel, nstates, m1, m2)


@njit
def apply_swap_kernel(state1, state2):
    return state2, state1

def apply_swap(state, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2, apply_swap_kernel, qubits)
