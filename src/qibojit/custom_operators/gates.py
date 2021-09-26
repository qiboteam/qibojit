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

		state[t0] = gate[0, 0] * buffer0 + gate[0, 1] * buffer1 + gate[0, 2] * buffer2 + gate[0, 3] * buffer3 + gate[0, 4] * buffer4 + gate[0, 5] * buffer5 + gate[0, 6] * buffer6 + gate[0, 7] * buffer7
		state[t1] = gate[1, 0] * buffer0 + gate[1, 1] * buffer1 + gate[1, 2] * buffer2 + gate[1, 3] * buffer3 + gate[1, 4] * buffer4 + gate[1, 5] * buffer5 + gate[1, 6] * buffer6 + gate[1, 7] * buffer7
		state[t2] = gate[2, 0] * buffer0 + gate[2, 1] * buffer1 + gate[2, 2] * buffer2 + gate[2, 3] * buffer3 + gate[2, 4] * buffer4 + gate[2, 5] * buffer5 + gate[2, 6] * buffer6 + gate[2, 7] * buffer7
		state[t3] = gate[3, 0] * buffer0 + gate[3, 1] * buffer1 + gate[3, 2] * buffer2 + gate[3, 3] * buffer3 + gate[3, 4] * buffer4 + gate[3, 5] * buffer5 + gate[3, 6] * buffer6 + gate[3, 7] * buffer7
		state[t4] = gate[4, 0] * buffer0 + gate[4, 1] * buffer1 + gate[4, 2] * buffer2 + gate[4, 3] * buffer3 + gate[4, 4] * buffer4 + gate[4, 5] * buffer5 + gate[4, 6] * buffer6 + gate[4, 7] * buffer7
		state[t5] = gate[5, 0] * buffer0 + gate[5, 1] * buffer1 + gate[5, 2] * buffer2 + gate[5, 3] * buffer3 + gate[5, 4] * buffer4 + gate[5, 5] * buffer5 + gate[5, 6] * buffer6 + gate[5, 7] * buffer7
		state[t6] = gate[6, 0] * buffer0 + gate[6, 1] * buffer1 + gate[6, 2] * buffer2 + gate[6, 3] * buffer3 + gate[6, 4] * buffer4 + gate[6, 5] * buffer5 + gate[6, 6] * buffer6 + gate[6, 7] * buffer7
		state[t7] = gate[7, 0] * buffer0 + gate[7, 1] * buffer1 + gate[7, 2] * buffer2 + gate[7, 3] * buffer3 + gate[7, 4] * buffer4 + gate[7, 5] * buffer5 + gate[7, 6] * buffer6 + gate[7, 7] * buffer7
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

		state[t0] = gate[0, 0] * buffer0 + gate[0, 1] * buffer1 + gate[0, 2] * buffer2 + gate[0, 3] * buffer3 + gate[0, 4] * buffer4 + gate[0, 5] * buffer5 + gate[0, 6] * buffer6 + gate[0, 7] * buffer7 + gate[0, 8] * buffer8 + gate[0, 9] * buffer9 + gate[0, 10] * buffer10 + gate[0, 11] * buffer11 + gate[0, 12] * buffer12 + gate[0, 13] * buffer13 + gate[0, 14] * buffer14 + gate[0, 15] * buffer15
		state[t1] = gate[1, 0] * buffer0 + gate[1, 1] * buffer1 + gate[1, 2] * buffer2 + gate[1, 3] * buffer3 + gate[1, 4] * buffer4 + gate[1, 5] * buffer5 + gate[1, 6] * buffer6 + gate[1, 7] * buffer7 + gate[1, 8] * buffer8 + gate[1, 9] * buffer9 + gate[1, 10] * buffer10 + gate[1, 11] * buffer11 + gate[1, 12] * buffer12 + gate[1, 13] * buffer13 + gate[1, 14] * buffer14 + gate[1, 15] * buffer15
		state[t2] = gate[2, 0] * buffer0 + gate[2, 1] * buffer1 + gate[2, 2] * buffer2 + gate[2, 3] * buffer3 + gate[2, 4] * buffer4 + gate[2, 5] * buffer5 + gate[2, 6] * buffer6 + gate[2, 7] * buffer7 + gate[2, 8] * buffer8 + gate[2, 9] * buffer9 + gate[2, 10] * buffer10 + gate[2, 11] * buffer11 + gate[2, 12] * buffer12 + gate[2, 13] * buffer13 + gate[2, 14] * buffer14 + gate[2, 15] * buffer15
		state[t3] = gate[3, 0] * buffer0 + gate[3, 1] * buffer1 + gate[3, 2] * buffer2 + gate[3, 3] * buffer3 + gate[3, 4] * buffer4 + gate[3, 5] * buffer5 + gate[3, 6] * buffer6 + gate[3, 7] * buffer7 + gate[3, 8] * buffer8 + gate[3, 9] * buffer9 + gate[3, 10] * buffer10 + gate[3, 11] * buffer11 + gate[3, 12] * buffer12 + gate[3, 13] * buffer13 + gate[3, 14] * buffer14 + gate[3, 15] * buffer15
		state[t4] = gate[4, 0] * buffer0 + gate[4, 1] * buffer1 + gate[4, 2] * buffer2 + gate[4, 3] * buffer3 + gate[4, 4] * buffer4 + gate[4, 5] * buffer5 + gate[4, 6] * buffer6 + gate[4, 7] * buffer7 + gate[4, 8] * buffer8 + gate[4, 9] * buffer9 + gate[4, 10] * buffer10 + gate[4, 11] * buffer11 + gate[4, 12] * buffer12 + gate[4, 13] * buffer13 + gate[4, 14] * buffer14 + gate[4, 15] * buffer15
		state[t5] = gate[5, 0] * buffer0 + gate[5, 1] * buffer1 + gate[5, 2] * buffer2 + gate[5, 3] * buffer3 + gate[5, 4] * buffer4 + gate[5, 5] * buffer5 + gate[5, 6] * buffer6 + gate[5, 7] * buffer7 + gate[5, 8] * buffer8 + gate[5, 9] * buffer9 + gate[5, 10] * buffer10 + gate[5, 11] * buffer11 + gate[5, 12] * buffer12 + gate[5, 13] * buffer13 + gate[5, 14] * buffer14 + gate[5, 15] * buffer15
		state[t6] = gate[6, 0] * buffer0 + gate[6, 1] * buffer1 + gate[6, 2] * buffer2 + gate[6, 3] * buffer3 + gate[6, 4] * buffer4 + gate[6, 5] * buffer5 + gate[6, 6] * buffer6 + gate[6, 7] * buffer7 + gate[6, 8] * buffer8 + gate[6, 9] * buffer9 + gate[6, 10] * buffer10 + gate[6, 11] * buffer11 + gate[6, 12] * buffer12 + gate[6, 13] * buffer13 + gate[6, 14] * buffer14 + gate[6, 15] * buffer15
		state[t7] = gate[7, 0] * buffer0 + gate[7, 1] * buffer1 + gate[7, 2] * buffer2 + gate[7, 3] * buffer3 + gate[7, 4] * buffer4 + gate[7, 5] * buffer5 + gate[7, 6] * buffer6 + gate[7, 7] * buffer7 + gate[7, 8] * buffer8 + gate[7, 9] * buffer9 + gate[7, 10] * buffer10 + gate[7, 11] * buffer11 + gate[7, 12] * buffer12 + gate[7, 13] * buffer13 + gate[7, 14] * buffer14 + gate[7, 15] * buffer15
		state[t8] = gate[8, 0] * buffer0 + gate[8, 1] * buffer1 + gate[8, 2] * buffer2 + gate[8, 3] * buffer3 + gate[8, 4] * buffer4 + gate[8, 5] * buffer5 + gate[8, 6] * buffer6 + gate[8, 7] * buffer7 + gate[8, 8] * buffer8 + gate[8, 9] * buffer9 + gate[8, 10] * buffer10 + gate[8, 11] * buffer11 + gate[8, 12] * buffer12 + gate[8, 13] * buffer13 + gate[8, 14] * buffer14 + gate[8, 15] * buffer15
		state[t9] = gate[9, 0] * buffer0 + gate[9, 1] * buffer1 + gate[9, 2] * buffer2 + gate[9, 3] * buffer3 + gate[9, 4] * buffer4 + gate[9, 5] * buffer5 + gate[9, 6] * buffer6 + gate[9, 7] * buffer7 + gate[9, 8] * buffer8 + gate[9, 9] * buffer9 + gate[9, 10] * buffer10 + gate[9, 11] * buffer11 + gate[9, 12] * buffer12 + gate[9, 13] * buffer13 + gate[9, 14] * buffer14 + gate[9, 15] * buffer15
		state[t10] = gate[10, 0] * buffer0 + gate[10, 1] * buffer1 + gate[10, 2] * buffer2 + gate[10, 3] * buffer3 + gate[10, 4] * buffer4 + gate[10, 5] * buffer5 + gate[10, 6] * buffer6 + gate[10, 7] * buffer7 + gate[10, 8] * buffer8 + gate[10, 9] * buffer9 + gate[10, 10] * buffer10 + gate[10, 11] * buffer11 + gate[10, 12] * buffer12 + gate[10, 13] * buffer13 + gate[10, 14] * buffer14 + gate[10, 15] * buffer15
		state[t11] = gate[11, 0] * buffer0 + gate[11, 1] * buffer1 + gate[11, 2] * buffer2 + gate[11, 3] * buffer3 + gate[11, 4] * buffer4 + gate[11, 5] * buffer5 + gate[11, 6] * buffer6 + gate[11, 7] * buffer7 + gate[11, 8] * buffer8 + gate[11, 9] * buffer9 + gate[11, 10] * buffer10 + gate[11, 11] * buffer11 + gate[11, 12] * buffer12 + gate[11, 13] * buffer13 + gate[11, 14] * buffer14 + gate[11, 15] * buffer15
		state[t12] = gate[12, 0] * buffer0 + gate[12, 1] * buffer1 + gate[12, 2] * buffer2 + gate[12, 3] * buffer3 + gate[12, 4] * buffer4 + gate[12, 5] * buffer5 + gate[12, 6] * buffer6 + gate[12, 7] * buffer7 + gate[12, 8] * buffer8 + gate[12, 9] * buffer9 + gate[12, 10] * buffer10 + gate[12, 11] * buffer11 + gate[12, 12] * buffer12 + gate[12, 13] * buffer13 + gate[12, 14] * buffer14 + gate[12, 15] * buffer15
		state[t13] = gate[13, 0] * buffer0 + gate[13, 1] * buffer1 + gate[13, 2] * buffer2 + gate[13, 3] * buffer3 + gate[13, 4] * buffer4 + gate[13, 5] * buffer5 + gate[13, 6] * buffer6 + gate[13, 7] * buffer7 + gate[13, 8] * buffer8 + gate[13, 9] * buffer9 + gate[13, 10] * buffer10 + gate[13, 11] * buffer11 + gate[13, 12] * buffer12 + gate[13, 13] * buffer13 + gate[13, 14] * buffer14 + gate[13, 15] * buffer15
		state[t14] = gate[14, 0] * buffer0 + gate[14, 1] * buffer1 + gate[14, 2] * buffer2 + gate[14, 3] * buffer3 + gate[14, 4] * buffer4 + gate[14, 5] * buffer5 + gate[14, 6] * buffer6 + gate[14, 7] * buffer7 + gate[14, 8] * buffer8 + gate[14, 9] * buffer9 + gate[14, 10] * buffer10 + gate[14, 11] * buffer11 + gate[14, 12] * buffer12 + gate[14, 13] * buffer13 + gate[14, 14] * buffer14 + gate[14, 15] * buffer15
		state[t15] = gate[15, 0] * buffer0 + gate[15, 1] * buffer1 + gate[15, 2] * buffer2 + gate[15, 3] * buffer3 + gate[15, 4] * buffer4 + gate[15, 5] * buffer5 + gate[15, 6] * buffer6 + gate[15, 7] * buffer7 + gate[15, 8] * buffer8 + gate[15, 9] * buffer9 + gate[15, 10] * buffer10 + gate[15, 11] * buffer11 + gate[15, 12] * buffer12 + gate[15, 13] * buffer13 + gate[15, 14] * buffer14 + gate[15, 15] * buffer15
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

		state[t0] = gate[0, 0] * buffer0 + gate[0, 1] * buffer1 + gate[0, 2] * buffer2 + gate[0, 3] * buffer3 + gate[0, 4] * buffer4 + gate[0, 5] * buffer5 + gate[0, 6] * buffer6 + gate[0, 7] * buffer7 + gate[0, 8] * buffer8 + gate[0, 9] * buffer9 + gate[0, 10] * buffer10 + gate[0, 11] * buffer11 + gate[0, 12] * buffer12 + gate[0, 13] * buffer13 + gate[0, 14] * buffer14 + gate[0, 15] * buffer15 + gate[0, 16] * buffer16 + gate[0, 17] * buffer17 + gate[0, 18] * buffer18 + gate[0, 19] * buffer19 + gate[0, 20] * buffer20 + gate[0, 21] * buffer21 + gate[0, 22] * buffer22 + gate[0, 23] * buffer23 + gate[0, 24] * buffer24 + gate[0, 25] * buffer25 + gate[0, 26] * buffer26 + gate[0, 27] * buffer27 + gate[0, 28] * buffer28 + gate[0, 29] * buffer29 + gate[0, 30] * buffer30 + gate[0, 31] * buffer31
		state[t1] = gate[1, 0] * buffer0 + gate[1, 1] * buffer1 + gate[1, 2] * buffer2 + gate[1, 3] * buffer3 + gate[1, 4] * buffer4 + gate[1, 5] * buffer5 + gate[1, 6] * buffer6 + gate[1, 7] * buffer7 + gate[1, 8] * buffer8 + gate[1, 9] * buffer9 + gate[1, 10] * buffer10 + gate[1, 11] * buffer11 + gate[1, 12] * buffer12 + gate[1, 13] * buffer13 + gate[1, 14] * buffer14 + gate[1, 15] * buffer15 + gate[1, 16] * buffer16 + gate[1, 17] * buffer17 + gate[1, 18] * buffer18 + gate[1, 19] * buffer19 + gate[1, 20] * buffer20 + gate[1, 21] * buffer21 + gate[1, 22] * buffer22 + gate[1, 23] * buffer23 + gate[1, 24] * buffer24 + gate[1, 25] * buffer25 + gate[1, 26] * buffer26 + gate[1, 27] * buffer27 + gate[1, 28] * buffer28 + gate[1, 29] * buffer29 + gate[1, 30] * buffer30 + gate[1, 31] * buffer31
		state[t2] = gate[2, 0] * buffer0 + gate[2, 1] * buffer1 + gate[2, 2] * buffer2 + gate[2, 3] * buffer3 + gate[2, 4] * buffer4 + gate[2, 5] * buffer5 + gate[2, 6] * buffer6 + gate[2, 7] * buffer7 + gate[2, 8] * buffer8 + gate[2, 9] * buffer9 + gate[2, 10] * buffer10 + gate[2, 11] * buffer11 + gate[2, 12] * buffer12 + gate[2, 13] * buffer13 + gate[2, 14] * buffer14 + gate[2, 15] * buffer15 + gate[2, 16] * buffer16 + gate[2, 17] * buffer17 + gate[2, 18] * buffer18 + gate[2, 19] * buffer19 + gate[2, 20] * buffer20 + gate[2, 21] * buffer21 + gate[2, 22] * buffer22 + gate[2, 23] * buffer23 + gate[2, 24] * buffer24 + gate[2, 25] * buffer25 + gate[2, 26] * buffer26 + gate[2, 27] * buffer27 + gate[2, 28] * buffer28 + gate[2, 29] * buffer29 + gate[2, 30] * buffer30 + gate[2, 31] * buffer31
		state[t3] = gate[3, 0] * buffer0 + gate[3, 1] * buffer1 + gate[3, 2] * buffer2 + gate[3, 3] * buffer3 + gate[3, 4] * buffer4 + gate[3, 5] * buffer5 + gate[3, 6] * buffer6 + gate[3, 7] * buffer7 + gate[3, 8] * buffer8 + gate[3, 9] * buffer9 + gate[3, 10] * buffer10 + gate[3, 11] * buffer11 + gate[3, 12] * buffer12 + gate[3, 13] * buffer13 + gate[3, 14] * buffer14 + gate[3, 15] * buffer15 + gate[3, 16] * buffer16 + gate[3, 17] * buffer17 + gate[3, 18] * buffer18 + gate[3, 19] * buffer19 + gate[3, 20] * buffer20 + gate[3, 21] * buffer21 + gate[3, 22] * buffer22 + gate[3, 23] * buffer23 + gate[3, 24] * buffer24 + gate[3, 25] * buffer25 + gate[3, 26] * buffer26 + gate[3, 27] * buffer27 + gate[3, 28] * buffer28 + gate[3, 29] * buffer29 + gate[3, 30] * buffer30 + gate[3, 31] * buffer31
		state[t4] = gate[4, 0] * buffer0 + gate[4, 1] * buffer1 + gate[4, 2] * buffer2 + gate[4, 3] * buffer3 + gate[4, 4] * buffer4 + gate[4, 5] * buffer5 + gate[4, 6] * buffer6 + gate[4, 7] * buffer7 + gate[4, 8] * buffer8 + gate[4, 9] * buffer9 + gate[4, 10] * buffer10 + gate[4, 11] * buffer11 + gate[4, 12] * buffer12 + gate[4, 13] * buffer13 + gate[4, 14] * buffer14 + gate[4, 15] * buffer15 + gate[4, 16] * buffer16 + gate[4, 17] * buffer17 + gate[4, 18] * buffer18 + gate[4, 19] * buffer19 + gate[4, 20] * buffer20 + gate[4, 21] * buffer21 + gate[4, 22] * buffer22 + gate[4, 23] * buffer23 + gate[4, 24] * buffer24 + gate[4, 25] * buffer25 + gate[4, 26] * buffer26 + gate[4, 27] * buffer27 + gate[4, 28] * buffer28 + gate[4, 29] * buffer29 + gate[4, 30] * buffer30 + gate[4, 31] * buffer31
		state[t5] = gate[5, 0] * buffer0 + gate[5, 1] * buffer1 + gate[5, 2] * buffer2 + gate[5, 3] * buffer3 + gate[5, 4] * buffer4 + gate[5, 5] * buffer5 + gate[5, 6] * buffer6 + gate[5, 7] * buffer7 + gate[5, 8] * buffer8 + gate[5, 9] * buffer9 + gate[5, 10] * buffer10 + gate[5, 11] * buffer11 + gate[5, 12] * buffer12 + gate[5, 13] * buffer13 + gate[5, 14] * buffer14 + gate[5, 15] * buffer15 + gate[5, 16] * buffer16 + gate[5, 17] * buffer17 + gate[5, 18] * buffer18 + gate[5, 19] * buffer19 + gate[5, 20] * buffer20 + gate[5, 21] * buffer21 + gate[5, 22] * buffer22 + gate[5, 23] * buffer23 + gate[5, 24] * buffer24 + gate[5, 25] * buffer25 + gate[5, 26] * buffer26 + gate[5, 27] * buffer27 + gate[5, 28] * buffer28 + gate[5, 29] * buffer29 + gate[5, 30] * buffer30 + gate[5, 31] * buffer31
		state[t6] = gate[6, 0] * buffer0 + gate[6, 1] * buffer1 + gate[6, 2] * buffer2 + gate[6, 3] * buffer3 + gate[6, 4] * buffer4 + gate[6, 5] * buffer5 + gate[6, 6] * buffer6 + gate[6, 7] * buffer7 + gate[6, 8] * buffer8 + gate[6, 9] * buffer9 + gate[6, 10] * buffer10 + gate[6, 11] * buffer11 + gate[6, 12] * buffer12 + gate[6, 13] * buffer13 + gate[6, 14] * buffer14 + gate[6, 15] * buffer15 + gate[6, 16] * buffer16 + gate[6, 17] * buffer17 + gate[6, 18] * buffer18 + gate[6, 19] * buffer19 + gate[6, 20] * buffer20 + gate[6, 21] * buffer21 + gate[6, 22] * buffer22 + gate[6, 23] * buffer23 + gate[6, 24] * buffer24 + gate[6, 25] * buffer25 + gate[6, 26] * buffer26 + gate[6, 27] * buffer27 + gate[6, 28] * buffer28 + gate[6, 29] * buffer29 + gate[6, 30] * buffer30 + gate[6, 31] * buffer31
		state[t7] = gate[7, 0] * buffer0 + gate[7, 1] * buffer1 + gate[7, 2] * buffer2 + gate[7, 3] * buffer3 + gate[7, 4] * buffer4 + gate[7, 5] * buffer5 + gate[7, 6] * buffer6 + gate[7, 7] * buffer7 + gate[7, 8] * buffer8 + gate[7, 9] * buffer9 + gate[7, 10] * buffer10 + gate[7, 11] * buffer11 + gate[7, 12] * buffer12 + gate[7, 13] * buffer13 + gate[7, 14] * buffer14 + gate[7, 15] * buffer15 + gate[7, 16] * buffer16 + gate[7, 17] * buffer17 + gate[7, 18] * buffer18 + gate[7, 19] * buffer19 + gate[7, 20] * buffer20 + gate[7, 21] * buffer21 + gate[7, 22] * buffer22 + gate[7, 23] * buffer23 + gate[7, 24] * buffer24 + gate[7, 25] * buffer25 + gate[7, 26] * buffer26 + gate[7, 27] * buffer27 + gate[7, 28] * buffer28 + gate[7, 29] * buffer29 + gate[7, 30] * buffer30 + gate[7, 31] * buffer31
		state[t8] = gate[8, 0] * buffer0 + gate[8, 1] * buffer1 + gate[8, 2] * buffer2 + gate[8, 3] * buffer3 + gate[8, 4] * buffer4 + gate[8, 5] * buffer5 + gate[8, 6] * buffer6 + gate[8, 7] * buffer7 + gate[8, 8] * buffer8 + gate[8, 9] * buffer9 + gate[8, 10] * buffer10 + gate[8, 11] * buffer11 + gate[8, 12] * buffer12 + gate[8, 13] * buffer13 + gate[8, 14] * buffer14 + gate[8, 15] * buffer15 + gate[8, 16] * buffer16 + gate[8, 17] * buffer17 + gate[8, 18] * buffer18 + gate[8, 19] * buffer19 + gate[8, 20] * buffer20 + gate[8, 21] * buffer21 + gate[8, 22] * buffer22 + gate[8, 23] * buffer23 + gate[8, 24] * buffer24 + gate[8, 25] * buffer25 + gate[8, 26] * buffer26 + gate[8, 27] * buffer27 + gate[8, 28] * buffer28 + gate[8, 29] * buffer29 + gate[8, 30] * buffer30 + gate[8, 31] * buffer31
		state[t9] = gate[9, 0] * buffer0 + gate[9, 1] * buffer1 + gate[9, 2] * buffer2 + gate[9, 3] * buffer3 + gate[9, 4] * buffer4 + gate[9, 5] * buffer5 + gate[9, 6] * buffer6 + gate[9, 7] * buffer7 + gate[9, 8] * buffer8 + gate[9, 9] * buffer9 + gate[9, 10] * buffer10 + gate[9, 11] * buffer11 + gate[9, 12] * buffer12 + gate[9, 13] * buffer13 + gate[9, 14] * buffer14 + gate[9, 15] * buffer15 + gate[9, 16] * buffer16 + gate[9, 17] * buffer17 + gate[9, 18] * buffer18 + gate[9, 19] * buffer19 + gate[9, 20] * buffer20 + gate[9, 21] * buffer21 + gate[9, 22] * buffer22 + gate[9, 23] * buffer23 + gate[9, 24] * buffer24 + gate[9, 25] * buffer25 + gate[9, 26] * buffer26 + gate[9, 27] * buffer27 + gate[9, 28] * buffer28 + gate[9, 29] * buffer29 + gate[9, 30] * buffer30 + gate[9, 31] * buffer31
		state[t10] = gate[10, 0] * buffer0 + gate[10, 1] * buffer1 + gate[10, 2] * buffer2 + gate[10, 3] * buffer3 + gate[10, 4] * buffer4 + gate[10, 5] * buffer5 + gate[10, 6] * buffer6 + gate[10, 7] * buffer7 + gate[10, 8] * buffer8 + gate[10, 9] * buffer9 + gate[10, 10] * buffer10 + gate[10, 11] * buffer11 + gate[10, 12] * buffer12 + gate[10, 13] * buffer13 + gate[10, 14] * buffer14 + gate[10, 15] * buffer15 + gate[10, 16] * buffer16 + gate[10, 17] * buffer17 + gate[10, 18] * buffer18 + gate[10, 19] * buffer19 + gate[10, 20] * buffer20 + gate[10, 21] * buffer21 + gate[10, 22] * buffer22 + gate[10, 23] * buffer23 + gate[10, 24] * buffer24 + gate[10, 25] * buffer25 + gate[10, 26] * buffer26 + gate[10, 27] * buffer27 + gate[10, 28] * buffer28 + gate[10, 29] * buffer29 + gate[10, 30] * buffer30 + gate[10, 31] * buffer31
		state[t11] = gate[11, 0] * buffer0 + gate[11, 1] * buffer1 + gate[11, 2] * buffer2 + gate[11, 3] * buffer3 + gate[11, 4] * buffer4 + gate[11, 5] * buffer5 + gate[11, 6] * buffer6 + gate[11, 7] * buffer7 + gate[11, 8] * buffer8 + gate[11, 9] * buffer9 + gate[11, 10] * buffer10 + gate[11, 11] * buffer11 + gate[11, 12] * buffer12 + gate[11, 13] * buffer13 + gate[11, 14] * buffer14 + gate[11, 15] * buffer15 + gate[11, 16] * buffer16 + gate[11, 17] * buffer17 + gate[11, 18] * buffer18 + gate[11, 19] * buffer19 + gate[11, 20] * buffer20 + gate[11, 21] * buffer21 + gate[11, 22] * buffer22 + gate[11, 23] * buffer23 + gate[11, 24] * buffer24 + gate[11, 25] * buffer25 + gate[11, 26] * buffer26 + gate[11, 27] * buffer27 + gate[11, 28] * buffer28 + gate[11, 29] * buffer29 + gate[11, 30] * buffer30 + gate[11, 31] * buffer31
		state[t12] = gate[12, 0] * buffer0 + gate[12, 1] * buffer1 + gate[12, 2] * buffer2 + gate[12, 3] * buffer3 + gate[12, 4] * buffer4 + gate[12, 5] * buffer5 + gate[12, 6] * buffer6 + gate[12, 7] * buffer7 + gate[12, 8] * buffer8 + gate[12, 9] * buffer9 + gate[12, 10] * buffer10 + gate[12, 11] * buffer11 + gate[12, 12] * buffer12 + gate[12, 13] * buffer13 + gate[12, 14] * buffer14 + gate[12, 15] * buffer15 + gate[12, 16] * buffer16 + gate[12, 17] * buffer17 + gate[12, 18] * buffer18 + gate[12, 19] * buffer19 + gate[12, 20] * buffer20 + gate[12, 21] * buffer21 + gate[12, 22] * buffer22 + gate[12, 23] * buffer23 + gate[12, 24] * buffer24 + gate[12, 25] * buffer25 + gate[12, 26] * buffer26 + gate[12, 27] * buffer27 + gate[12, 28] * buffer28 + gate[12, 29] * buffer29 + gate[12, 30] * buffer30 + gate[12, 31] * buffer31
		state[t13] = gate[13, 0] * buffer0 + gate[13, 1] * buffer1 + gate[13, 2] * buffer2 + gate[13, 3] * buffer3 + gate[13, 4] * buffer4 + gate[13, 5] * buffer5 + gate[13, 6] * buffer6 + gate[13, 7] * buffer7 + gate[13, 8] * buffer8 + gate[13, 9] * buffer9 + gate[13, 10] * buffer10 + gate[13, 11] * buffer11 + gate[13, 12] * buffer12 + gate[13, 13] * buffer13 + gate[13, 14] * buffer14 + gate[13, 15] * buffer15 + gate[13, 16] * buffer16 + gate[13, 17] * buffer17 + gate[13, 18] * buffer18 + gate[13, 19] * buffer19 + gate[13, 20] * buffer20 + gate[13, 21] * buffer21 + gate[13, 22] * buffer22 + gate[13, 23] * buffer23 + gate[13, 24] * buffer24 + gate[13, 25] * buffer25 + gate[13, 26] * buffer26 + gate[13, 27] * buffer27 + gate[13, 28] * buffer28 + gate[13, 29] * buffer29 + gate[13, 30] * buffer30 + gate[13, 31] * buffer31
		state[t14] = gate[14, 0] * buffer0 + gate[14, 1] * buffer1 + gate[14, 2] * buffer2 + gate[14, 3] * buffer3 + gate[14, 4] * buffer4 + gate[14, 5] * buffer5 + gate[14, 6] * buffer6 + gate[14, 7] * buffer7 + gate[14, 8] * buffer8 + gate[14, 9] * buffer9 + gate[14, 10] * buffer10 + gate[14, 11] * buffer11 + gate[14, 12] * buffer12 + gate[14, 13] * buffer13 + gate[14, 14] * buffer14 + gate[14, 15] * buffer15 + gate[14, 16] * buffer16 + gate[14, 17] * buffer17 + gate[14, 18] * buffer18 + gate[14, 19] * buffer19 + gate[14, 20] * buffer20 + gate[14, 21] * buffer21 + gate[14, 22] * buffer22 + gate[14, 23] * buffer23 + gate[14, 24] * buffer24 + gate[14, 25] * buffer25 + gate[14, 26] * buffer26 + gate[14, 27] * buffer27 + gate[14, 28] * buffer28 + gate[14, 29] * buffer29 + gate[14, 30] * buffer30 + gate[14, 31] * buffer31
		state[t15] = gate[15, 0] * buffer0 + gate[15, 1] * buffer1 + gate[15, 2] * buffer2 + gate[15, 3] * buffer3 + gate[15, 4] * buffer4 + gate[15, 5] * buffer5 + gate[15, 6] * buffer6 + gate[15, 7] * buffer7 + gate[15, 8] * buffer8 + gate[15, 9] * buffer9 + gate[15, 10] * buffer10 + gate[15, 11] * buffer11 + gate[15, 12] * buffer12 + gate[15, 13] * buffer13 + gate[15, 14] * buffer14 + gate[15, 15] * buffer15 + gate[15, 16] * buffer16 + gate[15, 17] * buffer17 + gate[15, 18] * buffer18 + gate[15, 19] * buffer19 + gate[15, 20] * buffer20 + gate[15, 21] * buffer21 + gate[15, 22] * buffer22 + gate[15, 23] * buffer23 + gate[15, 24] * buffer24 + gate[15, 25] * buffer25 + gate[15, 26] * buffer26 + gate[15, 27] * buffer27 + gate[15, 28] * buffer28 + gate[15, 29] * buffer29 + gate[15, 30] * buffer30 + gate[15, 31] * buffer31
		state[t16] = gate[16, 0] * buffer0 + gate[16, 1] * buffer1 + gate[16, 2] * buffer2 + gate[16, 3] * buffer3 + gate[16, 4] * buffer4 + gate[16, 5] * buffer5 + gate[16, 6] * buffer6 + gate[16, 7] * buffer7 + gate[16, 8] * buffer8 + gate[16, 9] * buffer9 + gate[16, 10] * buffer10 + gate[16, 11] * buffer11 + gate[16, 12] * buffer12 + gate[16, 13] * buffer13 + gate[16, 14] * buffer14 + gate[16, 15] * buffer15 + gate[16, 16] * buffer16 + gate[16, 17] * buffer17 + gate[16, 18] * buffer18 + gate[16, 19] * buffer19 + gate[16, 20] * buffer20 + gate[16, 21] * buffer21 + gate[16, 22] * buffer22 + gate[16, 23] * buffer23 + gate[16, 24] * buffer24 + gate[16, 25] * buffer25 + gate[16, 26] * buffer26 + gate[16, 27] * buffer27 + gate[16, 28] * buffer28 + gate[16, 29] * buffer29 + gate[16, 30] * buffer30 + gate[16, 31] * buffer31
		state[t17] = gate[17, 0] * buffer0 + gate[17, 1] * buffer1 + gate[17, 2] * buffer2 + gate[17, 3] * buffer3 + gate[17, 4] * buffer4 + gate[17, 5] * buffer5 + gate[17, 6] * buffer6 + gate[17, 7] * buffer7 + gate[17, 8] * buffer8 + gate[17, 9] * buffer9 + gate[17, 10] * buffer10 + gate[17, 11] * buffer11 + gate[17, 12] * buffer12 + gate[17, 13] * buffer13 + gate[17, 14] * buffer14 + gate[17, 15] * buffer15 + gate[17, 16] * buffer16 + gate[17, 17] * buffer17 + gate[17, 18] * buffer18 + gate[17, 19] * buffer19 + gate[17, 20] * buffer20 + gate[17, 21] * buffer21 + gate[17, 22] * buffer22 + gate[17, 23] * buffer23 + gate[17, 24] * buffer24 + gate[17, 25] * buffer25 + gate[17, 26] * buffer26 + gate[17, 27] * buffer27 + gate[17, 28] * buffer28 + gate[17, 29] * buffer29 + gate[17, 30] * buffer30 + gate[17, 31] * buffer31
		state[t18] = gate[18, 0] * buffer0 + gate[18, 1] * buffer1 + gate[18, 2] * buffer2 + gate[18, 3] * buffer3 + gate[18, 4] * buffer4 + gate[18, 5] * buffer5 + gate[18, 6] * buffer6 + gate[18, 7] * buffer7 + gate[18, 8] * buffer8 + gate[18, 9] * buffer9 + gate[18, 10] * buffer10 + gate[18, 11] * buffer11 + gate[18, 12] * buffer12 + gate[18, 13] * buffer13 + gate[18, 14] * buffer14 + gate[18, 15] * buffer15 + gate[18, 16] * buffer16 + gate[18, 17] * buffer17 + gate[18, 18] * buffer18 + gate[18, 19] * buffer19 + gate[18, 20] * buffer20 + gate[18, 21] * buffer21 + gate[18, 22] * buffer22 + gate[18, 23] * buffer23 + gate[18, 24] * buffer24 + gate[18, 25] * buffer25 + gate[18, 26] * buffer26 + gate[18, 27] * buffer27 + gate[18, 28] * buffer28 + gate[18, 29] * buffer29 + gate[18, 30] * buffer30 + gate[18, 31] * buffer31
		state[t19] = gate[19, 0] * buffer0 + gate[19, 1] * buffer1 + gate[19, 2] * buffer2 + gate[19, 3] * buffer3 + gate[19, 4] * buffer4 + gate[19, 5] * buffer5 + gate[19, 6] * buffer6 + gate[19, 7] * buffer7 + gate[19, 8] * buffer8 + gate[19, 9] * buffer9 + gate[19, 10] * buffer10 + gate[19, 11] * buffer11 + gate[19, 12] * buffer12 + gate[19, 13] * buffer13 + gate[19, 14] * buffer14 + gate[19, 15] * buffer15 + gate[19, 16] * buffer16 + gate[19, 17] * buffer17 + gate[19, 18] * buffer18 + gate[19, 19] * buffer19 + gate[19, 20] * buffer20 + gate[19, 21] * buffer21 + gate[19, 22] * buffer22 + gate[19, 23] * buffer23 + gate[19, 24] * buffer24 + gate[19, 25] * buffer25 + gate[19, 26] * buffer26 + gate[19, 27] * buffer27 + gate[19, 28] * buffer28 + gate[19, 29] * buffer29 + gate[19, 30] * buffer30 + gate[19, 31] * buffer31
		state[t20] = gate[20, 0] * buffer0 + gate[20, 1] * buffer1 + gate[20, 2] * buffer2 + gate[20, 3] * buffer3 + gate[20, 4] * buffer4 + gate[20, 5] * buffer5 + gate[20, 6] * buffer6 + gate[20, 7] * buffer7 + gate[20, 8] * buffer8 + gate[20, 9] * buffer9 + gate[20, 10] * buffer10 + gate[20, 11] * buffer11 + gate[20, 12] * buffer12 + gate[20, 13] * buffer13 + gate[20, 14] * buffer14 + gate[20, 15] * buffer15 + gate[20, 16] * buffer16 + gate[20, 17] * buffer17 + gate[20, 18] * buffer18 + gate[20, 19] * buffer19 + gate[20, 20] * buffer20 + gate[20, 21] * buffer21 + gate[20, 22] * buffer22 + gate[20, 23] * buffer23 + gate[20, 24] * buffer24 + gate[20, 25] * buffer25 + gate[20, 26] * buffer26 + gate[20, 27] * buffer27 + gate[20, 28] * buffer28 + gate[20, 29] * buffer29 + gate[20, 30] * buffer30 + gate[20, 31] * buffer31
		state[t21] = gate[21, 0] * buffer0 + gate[21, 1] * buffer1 + gate[21, 2] * buffer2 + gate[21, 3] * buffer3 + gate[21, 4] * buffer4 + gate[21, 5] * buffer5 + gate[21, 6] * buffer6 + gate[21, 7] * buffer7 + gate[21, 8] * buffer8 + gate[21, 9] * buffer9 + gate[21, 10] * buffer10 + gate[21, 11] * buffer11 + gate[21, 12] * buffer12 + gate[21, 13] * buffer13 + gate[21, 14] * buffer14 + gate[21, 15] * buffer15 + gate[21, 16] * buffer16 + gate[21, 17] * buffer17 + gate[21, 18] * buffer18 + gate[21, 19] * buffer19 + gate[21, 20] * buffer20 + gate[21, 21] * buffer21 + gate[21, 22] * buffer22 + gate[21, 23] * buffer23 + gate[21, 24] * buffer24 + gate[21, 25] * buffer25 + gate[21, 26] * buffer26 + gate[21, 27] * buffer27 + gate[21, 28] * buffer28 + gate[21, 29] * buffer29 + gate[21, 30] * buffer30 + gate[21, 31] * buffer31
		state[t22] = gate[22, 0] * buffer0 + gate[22, 1] * buffer1 + gate[22, 2] * buffer2 + gate[22, 3] * buffer3 + gate[22, 4] * buffer4 + gate[22, 5] * buffer5 + gate[22, 6] * buffer6 + gate[22, 7] * buffer7 + gate[22, 8] * buffer8 + gate[22, 9] * buffer9 + gate[22, 10] * buffer10 + gate[22, 11] * buffer11 + gate[22, 12] * buffer12 + gate[22, 13] * buffer13 + gate[22, 14] * buffer14 + gate[22, 15] * buffer15 + gate[22, 16] * buffer16 + gate[22, 17] * buffer17 + gate[22, 18] * buffer18 + gate[22, 19] * buffer19 + gate[22, 20] * buffer20 + gate[22, 21] * buffer21 + gate[22, 22] * buffer22 + gate[22, 23] * buffer23 + gate[22, 24] * buffer24 + gate[22, 25] * buffer25 + gate[22, 26] * buffer26 + gate[22, 27] * buffer27 + gate[22, 28] * buffer28 + gate[22, 29] * buffer29 + gate[22, 30] * buffer30 + gate[22, 31] * buffer31
		state[t23] = gate[23, 0] * buffer0 + gate[23, 1] * buffer1 + gate[23, 2] * buffer2 + gate[23, 3] * buffer3 + gate[23, 4] * buffer4 + gate[23, 5] * buffer5 + gate[23, 6] * buffer6 + gate[23, 7] * buffer7 + gate[23, 8] * buffer8 + gate[23, 9] * buffer9 + gate[23, 10] * buffer10 + gate[23, 11] * buffer11 + gate[23, 12] * buffer12 + gate[23, 13] * buffer13 + gate[23, 14] * buffer14 + gate[23, 15] * buffer15 + gate[23, 16] * buffer16 + gate[23, 17] * buffer17 + gate[23, 18] * buffer18 + gate[23, 19] * buffer19 + gate[23, 20] * buffer20 + gate[23, 21] * buffer21 + gate[23, 22] * buffer22 + gate[23, 23] * buffer23 + gate[23, 24] * buffer24 + gate[23, 25] * buffer25 + gate[23, 26] * buffer26 + gate[23, 27] * buffer27 + gate[23, 28] * buffer28 + gate[23, 29] * buffer29 + gate[23, 30] * buffer30 + gate[23, 31] * buffer31
		state[t24] = gate[24, 0] * buffer0 + gate[24, 1] * buffer1 + gate[24, 2] * buffer2 + gate[24, 3] * buffer3 + gate[24, 4] * buffer4 + gate[24, 5] * buffer5 + gate[24, 6] * buffer6 + gate[24, 7] * buffer7 + gate[24, 8] * buffer8 + gate[24, 9] * buffer9 + gate[24, 10] * buffer10 + gate[24, 11] * buffer11 + gate[24, 12] * buffer12 + gate[24, 13] * buffer13 + gate[24, 14] * buffer14 + gate[24, 15] * buffer15 + gate[24, 16] * buffer16 + gate[24, 17] * buffer17 + gate[24, 18] * buffer18 + gate[24, 19] * buffer19 + gate[24, 20] * buffer20 + gate[24, 21] * buffer21 + gate[24, 22] * buffer22 + gate[24, 23] * buffer23 + gate[24, 24] * buffer24 + gate[24, 25] * buffer25 + gate[24, 26] * buffer26 + gate[24, 27] * buffer27 + gate[24, 28] * buffer28 + gate[24, 29] * buffer29 + gate[24, 30] * buffer30 + gate[24, 31] * buffer31
		state[t25] = gate[25, 0] * buffer0 + gate[25, 1] * buffer1 + gate[25, 2] * buffer2 + gate[25, 3] * buffer3 + gate[25, 4] * buffer4 + gate[25, 5] * buffer5 + gate[25, 6] * buffer6 + gate[25, 7] * buffer7 + gate[25, 8] * buffer8 + gate[25, 9] * buffer9 + gate[25, 10] * buffer10 + gate[25, 11] * buffer11 + gate[25, 12] * buffer12 + gate[25, 13] * buffer13 + gate[25, 14] * buffer14 + gate[25, 15] * buffer15 + gate[25, 16] * buffer16 + gate[25, 17] * buffer17 + gate[25, 18] * buffer18 + gate[25, 19] * buffer19 + gate[25, 20] * buffer20 + gate[25, 21] * buffer21 + gate[25, 22] * buffer22 + gate[25, 23] * buffer23 + gate[25, 24] * buffer24 + gate[25, 25] * buffer25 + gate[25, 26] * buffer26 + gate[25, 27] * buffer27 + gate[25, 28] * buffer28 + gate[25, 29] * buffer29 + gate[25, 30] * buffer30 + gate[25, 31] * buffer31
		state[t26] = gate[26, 0] * buffer0 + gate[26, 1] * buffer1 + gate[26, 2] * buffer2 + gate[26, 3] * buffer3 + gate[26, 4] * buffer4 + gate[26, 5] * buffer5 + gate[26, 6] * buffer6 + gate[26, 7] * buffer7 + gate[26, 8] * buffer8 + gate[26, 9] * buffer9 + gate[26, 10] * buffer10 + gate[26, 11] * buffer11 + gate[26, 12] * buffer12 + gate[26, 13] * buffer13 + gate[26, 14] * buffer14 + gate[26, 15] * buffer15 + gate[26, 16] * buffer16 + gate[26, 17] * buffer17 + gate[26, 18] * buffer18 + gate[26, 19] * buffer19 + gate[26, 20] * buffer20 + gate[26, 21] * buffer21 + gate[26, 22] * buffer22 + gate[26, 23] * buffer23 + gate[26, 24] * buffer24 + gate[26, 25] * buffer25 + gate[26, 26] * buffer26 + gate[26, 27] * buffer27 + gate[26, 28] * buffer28 + gate[26, 29] * buffer29 + gate[26, 30] * buffer30 + gate[26, 31] * buffer31
		state[t27] = gate[27, 0] * buffer0 + gate[27, 1] * buffer1 + gate[27, 2] * buffer2 + gate[27, 3] * buffer3 + gate[27, 4] * buffer4 + gate[27, 5] * buffer5 + gate[27, 6] * buffer6 + gate[27, 7] * buffer7 + gate[27, 8] * buffer8 + gate[27, 9] * buffer9 + gate[27, 10] * buffer10 + gate[27, 11] * buffer11 + gate[27, 12] * buffer12 + gate[27, 13] * buffer13 + gate[27, 14] * buffer14 + gate[27, 15] * buffer15 + gate[27, 16] * buffer16 + gate[27, 17] * buffer17 + gate[27, 18] * buffer18 + gate[27, 19] * buffer19 + gate[27, 20] * buffer20 + gate[27, 21] * buffer21 + gate[27, 22] * buffer22 + gate[27, 23] * buffer23 + gate[27, 24] * buffer24 + gate[27, 25] * buffer25 + gate[27, 26] * buffer26 + gate[27, 27] * buffer27 + gate[27, 28] * buffer28 + gate[27, 29] * buffer29 + gate[27, 30] * buffer30 + gate[27, 31] * buffer31
		state[t28] = gate[28, 0] * buffer0 + gate[28, 1] * buffer1 + gate[28, 2] * buffer2 + gate[28, 3] * buffer3 + gate[28, 4] * buffer4 + gate[28, 5] * buffer5 + gate[28, 6] * buffer6 + gate[28, 7] * buffer7 + gate[28, 8] * buffer8 + gate[28, 9] * buffer9 + gate[28, 10] * buffer10 + gate[28, 11] * buffer11 + gate[28, 12] * buffer12 + gate[28, 13] * buffer13 + gate[28, 14] * buffer14 + gate[28, 15] * buffer15 + gate[28, 16] * buffer16 + gate[28, 17] * buffer17 + gate[28, 18] * buffer18 + gate[28, 19] * buffer19 + gate[28, 20] * buffer20 + gate[28, 21] * buffer21 + gate[28, 22] * buffer22 + gate[28, 23] * buffer23 + gate[28, 24] * buffer24 + gate[28, 25] * buffer25 + gate[28, 26] * buffer26 + gate[28, 27] * buffer27 + gate[28, 28] * buffer28 + gate[28, 29] * buffer29 + gate[28, 30] * buffer30 + gate[28, 31] * buffer31
		state[t29] = gate[29, 0] * buffer0 + gate[29, 1] * buffer1 + gate[29, 2] * buffer2 + gate[29, 3] * buffer3 + gate[29, 4] * buffer4 + gate[29, 5] * buffer5 + gate[29, 6] * buffer6 + gate[29, 7] * buffer7 + gate[29, 8] * buffer8 + gate[29, 9] * buffer9 + gate[29, 10] * buffer10 + gate[29, 11] * buffer11 + gate[29, 12] * buffer12 + gate[29, 13] * buffer13 + gate[29, 14] * buffer14 + gate[29, 15] * buffer15 + gate[29, 16] * buffer16 + gate[29, 17] * buffer17 + gate[29, 18] * buffer18 + gate[29, 19] * buffer19 + gate[29, 20] * buffer20 + gate[29, 21] * buffer21 + gate[29, 22] * buffer22 + gate[29, 23] * buffer23 + gate[29, 24] * buffer24 + gate[29, 25] * buffer25 + gate[29, 26] * buffer26 + gate[29, 27] * buffer27 + gate[29, 28] * buffer28 + gate[29, 29] * buffer29 + gate[29, 30] * buffer30 + gate[29, 31] * buffer31
		state[t30] = gate[30, 0] * buffer0 + gate[30, 1] * buffer1 + gate[30, 2] * buffer2 + gate[30, 3] * buffer3 + gate[30, 4] * buffer4 + gate[30, 5] * buffer5 + gate[30, 6] * buffer6 + gate[30, 7] * buffer7 + gate[30, 8] * buffer8 + gate[30, 9] * buffer9 + gate[30, 10] * buffer10 + gate[30, 11] * buffer11 + gate[30, 12] * buffer12 + gate[30, 13] * buffer13 + gate[30, 14] * buffer14 + gate[30, 15] * buffer15 + gate[30, 16] * buffer16 + gate[30, 17] * buffer17 + gate[30, 18] * buffer18 + gate[30, 19] * buffer19 + gate[30, 20] * buffer20 + gate[30, 21] * buffer21 + gate[30, 22] * buffer22 + gate[30, 23] * buffer23 + gate[30, 24] * buffer24 + gate[30, 25] * buffer25 + gate[30, 26] * buffer26 + gate[30, 27] * buffer27 + gate[30, 28] * buffer28 + gate[30, 29] * buffer29 + gate[30, 30] * buffer30 + gate[30, 31] * buffer31
		state[t31] = gate[31, 0] * buffer0 + gate[31, 1] * buffer1 + gate[31, 2] * buffer2 + gate[31, 3] * buffer3 + gate[31, 4] * buffer4 + gate[31, 5] * buffer5 + gate[31, 6] * buffer6 + gate[31, 7] * buffer7 + gate[31, 8] * buffer8 + gate[31, 9] * buffer9 + gate[31, 10] * buffer10 + gate[31, 11] * buffer11 + gate[31, 12] * buffer12 + gate[31, 13] * buffer13 + gate[31, 14] * buffer14 + gate[31, 15] * buffer15 + gate[31, 16] * buffer16 + gate[31, 17] * buffer17 + gate[31, 18] * buffer18 + gate[31, 19] * buffer19 + gate[31, 20] * buffer20 + gate[31, 21] * buffer21 + gate[31, 22] * buffer22 + gate[31, 23] * buffer23 + gate[31, 24] * buffer24 + gate[31, 25] * buffer25 + gate[31, 26] * buffer26 + gate[31, 27] * buffer27 + gate[31, 28] * buffer28 + gate[31, 29] * buffer29 + gate[31, 30] * buffer30 + gate[31, 31] * buffer31
	return state
