import numpy as np
from numba import prange, njit


@njit(cache=True)
def multicontrol_index(g, qubits):
    i = 0
    i += g
    for n in qubits:
        k = 1 << n
        i = ((i >> n) << (n + 1)) + (i & (k - 1)) + k
    return i


@njit(parallel=True, cache=True)
def apply_gate_kernel(state, gate, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i1 = ((g >> m) << (m + 1)) + (g & (tk - 1))
        i2 = i1 + tk
        state[i1], state[i2] = (gate[0, 0] * state[i1] + gate[0, 1] * state[i2],
                                gate[1, 0] * state[i1] + gate[1, 1] * state[i2])
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_gate_kernel(state, gate, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - tk, i
        state[i1], state[i2] = (gate[0, 0] * state[i1] + gate[0, 1] * state[i2],
                                gate[1, 0] * state[i1] + gate[1, 1] * state[i2])
    return state


@njit(parallel=True, cache=True)
def apply_x_kernel(state, gate, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i1 = ((g >> m) << (m + 1)) + (g & (tk - 1))
        i2 = i1 + tk
        state[i1], state[i2] = state[i2], state[i1]
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_x_kernel(state, gate, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - tk, i
        state[i1], state[i2] = state[i2], state[i1]
    return state


@njit(parallel=True, cache=True)
def apply_y_kernel(state, gate, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i1 = ((g >> m) << (m + 1)) + (g & (tk - 1))
        i2 = i1 + tk
        state[i1], state[i2] = -1j * state[i2], 1j * state[i1]
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_y_kernel(state, gate, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - tk, i
        state[i1], state[i2] = -1j * state[i2], 1j * state[i1]
    return state


@njit(parallel=True, cache=True)
def apply_z_kernel(state, gate, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m) << (m + 1)) + (g & (tk - 1))
        state[i + tk] *= -1
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_z_kernel(state, gate, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        state[i] *= -1
    return state


@njit(parallel=True, cache=True)
def apply_z_pow_kernel(state, gate, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m) << (m + 1)) + (g & (tk - 1))
        state[i + tk] = gate * state[i + tk]
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_z_pow_kernel(state, gate, qubits, nstates, m):
    tk = 1 << m
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        state[i] = gate * state[i]
    return state


@njit(parallel=True, cache=True)
def apply_two_qubit_gate_kernel(state, gate, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m1) << (m1 + 1)) + (g & (tk1 - 1))
        i = ((i >> m2) << (m2 + 1)) + (i & (tk2 - 1))
        i1, i2 = i + uk1, i + uk2
        i3 = i + tk1 + tk2
        buffer0, buffer1, buffer2 = state[i], state[i1], state[i2]
        state[i] = (gate[0, 0] * state[i] + gate[0, 1] * state[i1] +
                    gate[0, 2] * state[i2] + gate[0, 3] * state[i3])
        state[i1] = (gate[1, 0] * buffer0 + gate[1, 1] * state[i1] +
                     gate[1, 2] * state[i2] + gate[1, 3] * state[i3])
        state[i2] = (gate[2, 0] * buffer0 + gate[2, 1] * buffer1 +
                     gate[2, 2] * state[i2] + gate[2, 3] * state[i3])
        state[i3] = (gate[3, 0] * buffer0 + gate[3, 1] * buffer1 +
                     gate[3, 2] * buffer2 + gate[3, 3] * state[i3])
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_two_qubit_gate_kernel(state, gate, qubits, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - uk2, i - uk1
        i0 = i1 - uk1
        buffer0, buffer1, buffer2 = state[i0], state[i1], state[i2]
        state[i0] = (gate[0, 0] * state[i0] + gate[0, 1] * state[i1] +
                     gate[0, 2] * state[i2] + gate[0, 3] * state[i])
        state[i1] = (gate[1, 0] * buffer0 + gate[1, 1] * state[i1] +
                     gate[1, 2] * state[i2] + gate[1, 3] * state[i])
        state[i2] = (gate[2, 0] * buffer0 + gate[2, 1] * buffer1 +
                     gate[2, 2] * state[i2] + gate[2, 3] * state[i])
        state[i] = (gate[3, 0] * buffer0 + gate[3, 1] * buffer1 +
                    gate[3, 2] * buffer2 + gate[3, 3] * state[i])
    return state


@njit(parallel=True, cache=True)
def apply_swap_kernel(state, gate, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m1) << (m1 + 1)) + (g & (tk1 - 1))
        i = ((i >> m2) << (m2 + 1)) + (i & (tk2 - 1))
        i1, i2 = i + tk1, i + tk2
        state[i1], state[i2] = state[i2], state[i1]
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_swap_kernel(state, gate, qubits, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - tk2, i - tk1
        state[i1], state[i2] = state[i2], state[i1]
    return state


@njit(parallel=True, cache=True)
def apply_fsim_kernel(state, gate, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = ((g >> m1) << (m1 + 1)) + (g & (tk1 - 1))
        i = ((i >> m2) << (m2 + 1)) + (i & (tk2 - 1))
        i1, i2 = i + uk1, i + uk2
        i3 = i + tk1 + tk2
        state[i1], state[i2] = (gate[0] * state[i1] + gate[1] * state[i2],
                                gate[2] * state[i1] + gate[3] * state[i2])
        state[i3] *= gate[4]
    return state


@njit(parallel=True, cache=True)
def multicontrol_apply_fsim_kernel(state, gate, qubits, nstates, m1, m2, swap_targets=False):
    tk1, tk2 = 1 << m1, 1 << m2
    uk1, uk2 = tk1, tk2
    if swap_targets:
        uk1, uk2 = uk2, uk1
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        i = multicontrol_index(g, qubits)
        i1, i2 = i - uk2, i - uk1
        state[i1], state[i2] = (gate[0] * state[i1] + gate[1] * state[i2],
                                gate[2] * state[i1] + gate[3] * state[i2])
        state[i] *= gate[4]
    return state


@njit(cache=True)
def multitarget_index(i, targets):
    t = 0
    for u, v in enumerate(targets):
        t += ((i >> u) & 1) << v
    return t


@njit(parallel=True, cache=True)
def apply_multiqubit_gate_kernel(state, gate, qubits, nstates, targets, total):
    nsubstates = 1 << len(targets)
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        ig = multicontrol_index(g, qubits) - total
        buffer = np.empty(nsubstates, dtype=state.dtype)
        for i in range(nsubstates):
            t = ig + multitarget_index(i, targets)
            buffer[i] = state[t]
            state[t] = 0
            for j in range(min(i + 1, nsubstates)):
                state[t] += gate[i, j] * buffer[j]
            for j in range(i + 1, nsubstates):
                s = ig + multitarget_index(j, targets)
                state[t] += gate[i, j] * state[s]
    return state
