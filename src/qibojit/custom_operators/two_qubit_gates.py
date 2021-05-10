import numpy as np
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
        substate = np.array([state[i], state[i1], state[i2], state[i3]])
        state[i], state[i1], state[i2], state[i3] = kernel(substate, gate)
    return state


@njit(parallel=True)
def multicontrol_apply(state, gate, kernel, qubits, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    ncontrols = len(qubits)
    for g in prange(nstates):
        i = 0
        i += g
        for m in qubits:
            k = 1 << m
            i = ((i >> m) << (m + 1)) + (i & (k - 1)) + k
        i1, i2 = i - uk2, i - uk1
        i0 = i1 - uk1
        substate = np.array([state[i0], state[i1], state[i2], state[i]])
        state[i0], state[i1], state[i2], state[i] = kernel(substate, gate)
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
def apply_two_qubit_gate_kernel(substate, gate):
    return gate.dot(substate)

def apply_two_qubit_gate(state, gate, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2,
                           apply_two_qubit_gate_kernel,
                           qubits, gate)


@njit
def apply_swap_kernel(substate, gate):
    return substate[0], substate[2], substate[1], substate[3]

def apply_swap(state, nqubits, target1, target2, qubits=None):
    return apply_gate_base(state, nqubits, target1, target2, apply_swap_kernel,
                           qubits)
