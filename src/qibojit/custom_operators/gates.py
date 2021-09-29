from numba import prange, njit
from types import FunctionType


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
def apply_three_qubit_gate_kernel(state, gate, qubits, nstates, targets):
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        ig = multicontrol_index(g, qubits)
        t0 = ig - targets[0] - targets[1] - targets[2]
        buffer0 = state[t0]
        t1 = ig - targets[1] - targets[2]
        buffer1 = state[t1]
        t2 = ig - targets[0] - targets[2]
        buffer2 = state[t2]
        t3 = ig - targets[2]
        buffer3 = state[t3]
        t4 = ig - targets[0] - targets[1]
        buffer4 = state[t4]
        t5 = ig - targets[1]
        buffer5 = state[t5]
        t6 = ig - targets[0]
        buffer6 = state[t6]
        t7 = ig
        buffer7 = state[t7]

        for i in range(8):
            t = ig - multitarget_index(7 - i, targets)
            state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7
    return state


@njit(parallel=True, cache=True)
def apply_four_qubit_gate_kernel(state, gate, qubits, nstates, targets):
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        ig = multicontrol_index(g, qubits)
        t0 = ig - targets[0] - targets[1] - targets[2] - targets[3]
        buffer0 = state[t0]
        t1 = ig - targets[1] - targets[2] - targets[3]
        buffer1 = state[t1]
        t2 = ig - targets[0] - targets[2] - targets[3]
        buffer2 = state[t2]
        t3 = ig - targets[2] - targets[3]
        buffer3 = state[t3]
        t4 = ig - targets[0] - targets[1] - targets[3]
        buffer4 = state[t4]
        t5 = ig - targets[1] - targets[3]
        buffer5 = state[t5]
        t6 = ig - targets[0] - targets[3]
        buffer6 = state[t6]
        t7 = ig - targets[3]
        buffer7 = state[t7]
        t8 = ig - targets[0] - targets[1] - targets[2]
        buffer8 = state[t8]
        t9 = ig - targets[1] - targets[2]
        buffer9 = state[t9]
        t10 = ig - targets[0] - targets[2]
        buffer10 = state[t10]
        t11 = ig - targets[2]
        buffer11 = state[t11]
        t12 = ig - targets[0] - targets[1]
        buffer12 = state[t12]
        t13 = ig - targets[1]
        buffer13 = state[t13]
        t14 = ig - targets[0]
        buffer14 = state[t14]
        t15 = ig
        buffer15 = state[t15]

        for i in range(16):
            t = ig - multitarget_index(15 - i, targets)
            state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15
    return state


@njit(parallel=True, cache=True)
def apply_five_qubit_gate_kernel(state, gate, qubits, nstates, targets):
    for g in prange(nstates):  # pylint: disable=not-an-iterable
        ig = multicontrol_index(g, qubits)
        t0 = ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4]
        buffer0 = state[t0]
        t1 = ig - targets[1] - targets[2] - targets[3] - targets[4]
        buffer1 = state[t1]
        t2 = ig - targets[0] - targets[2] - targets[3] - targets[4]
        buffer2 = state[t2]
        t3 = ig - targets[2] - targets[3] - targets[4]
        buffer3 = state[t3]
        t4 = ig - targets[0] - targets[1] - targets[3] - targets[4]
        buffer4 = state[t4]
        t5 = ig - targets[1] - targets[3] - targets[4]
        buffer5 = state[t5]
        t6 = ig - targets[0] - targets[3] - targets[4]
        buffer6 = state[t6]
        t7 = ig - targets[3] - targets[4]
        buffer7 = state[t7]
        t8 = ig - targets[0] - targets[1] - targets[2] - targets[4]
        buffer8 = state[t8]
        t9 = ig - targets[1] - targets[2] - targets[4]
        buffer9 = state[t9]
        t10 = ig - targets[0] - targets[2] - targets[4]
        buffer10 = state[t10]
        t11 = ig - targets[2] - targets[4]
        buffer11 = state[t11]
        t12 = ig - targets[0] - targets[1] - targets[4]
        buffer12 = state[t12]
        t13 = ig - targets[1] - targets[4]
        buffer13 = state[t13]
        t14 = ig - targets[0] - targets[4]
        buffer14 = state[t14]
        t15 = ig - targets[4]
        buffer15 = state[t15]
        t16 = ig - targets[0] - targets[1] - targets[2] - targets[3]
        buffer16 = state[t16]
        t17 = ig - targets[1] - targets[2] - targets[3]
        buffer17 = state[t17]
        t18 = ig - targets[0] - targets[2] - targets[3]
        buffer18 = state[t18]
        t19 = ig - targets[2] - targets[3]
        buffer19 = state[t19]
        t20 = ig - targets[0] - targets[1] - targets[3]
        buffer20 = state[t20]
        t21 = ig - targets[1] - targets[3]
        buffer21 = state[t21]
        t22 = ig - targets[0] - targets[3]
        buffer22 = state[t22]
        t23 = ig - targets[3]
        buffer23 = state[t23]
        t24 = ig - targets[0] - targets[1] - targets[2]
        buffer24 = state[t24]
        t25 = ig - targets[1] - targets[2]
        buffer25 = state[t25]
        t26 = ig - targets[0] - targets[2]
        buffer26 = state[t26]
        t27 = ig - targets[2]
        buffer27 = state[t27]
        t28 = ig - targets[0] - targets[1]
        buffer28 = state[t28]
        t29 = ig - targets[1]
        buffer29 = state[t29]
        t30 = ig - targets[0]
        buffer30 = state[t30]
        t31 = ig
        buffer31 = state[t31]

        for i in range(32):
            t = ig - multitarget_index(31 - i, targets)
            state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15 + gate[i, 16] * buffer16 + gate[i, 17] * buffer17 + gate[i, 18] * buffer18 + gate[i, 19] * buffer19 + gate[i, 20] * buffer20 + gate[i, 21] * buffer21 + gate[i, 22] * buffer22 + gate[i, 23] * buffer23 + gate[i, 24] * buffer24 + gate[i, 25] * buffer25 + gate[i, 26] * buffer26 + gate[i, 27] * buffer27 + gate[i, 28] * buffer28 + gate[i, 29] * buffer29 + gate[i, 30] * buffer30 + gate[i, 31] * buffer31
    return state
