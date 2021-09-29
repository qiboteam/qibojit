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
        t += ((i >> u) & 1) * v
    return t


@njit(parallel=True, cache=True)
def apply_three_qubit_gate_kernel(state, gate, qubits, nstates, targets):
	for g in prange(nstates):  # pylint: disable=not-an-iterable
		ig = multicontrol_index(g, qubits)
		buffer0 = state[ig - targets[0] - targets[1] - targets[2]]
		buffer1 = state[ig - targets[1] - targets[2]]
		buffer2 = state[ig - targets[0] - targets[2]]
		buffer3 = state[ig - targets[2]]
		buffer4 = state[ig - targets[0] - targets[1]]
		buffer5 = state[ig - targets[1]]
		buffer6 = state[ig - targets[0]]
		buffer7 = state[ig]
		for i in range(8):
			t = ig - multitarget_index(7 - i, targets)
			state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7
	return state


@njit(parallel=True, cache=True)
def apply_four_qubit_gate_kernel(state, gate, qubits, nstates, targets):
	for g in prange(nstates):  # pylint: disable=not-an-iterable
		ig = multicontrol_index(g, qubits)
		buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]]
		buffer1 = state[ig - targets[1] - targets[2] - targets[3]]
		buffer2 = state[ig - targets[0] - targets[2] - targets[3]]
		buffer3 = state[ig - targets[2] - targets[3]]
		buffer4 = state[ig - targets[0] - targets[1] - targets[3]]
		buffer5 = state[ig - targets[1] - targets[3]]
		buffer6 = state[ig - targets[0] - targets[3]]
		buffer7 = state[ig - targets[3]]
		buffer8 = state[ig - targets[0] - targets[1] - targets[2]]
		buffer9 = state[ig - targets[1] - targets[2]]
		buffer10 = state[ig - targets[0] - targets[2]]
		buffer11 = state[ig - targets[2]]
		buffer12 = state[ig - targets[0] - targets[1]]
		buffer13 = state[ig - targets[1]]
		buffer14 = state[ig - targets[0]]
		buffer15 = state[ig]
		for i in range(16):
			t = ig - multitarget_index(15 - i, targets)
			state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15
	return state


@njit(parallel=True, cache=True)
def apply_five_qubit_gate_kernel(state, gate, qubits, nstates, targets):
	for g in prange(nstates):  # pylint: disable=not-an-iterable
		ig = multicontrol_index(g, qubits)
		buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer1 = state[ig - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer2 = state[ig - targets[0] - targets[2] - targets[3] - targets[4]]
		buffer3 = state[ig - targets[2] - targets[3] - targets[4]]
		buffer4 = state[ig - targets[0] - targets[1] - targets[3] - targets[4]]
		buffer5 = state[ig - targets[1] - targets[3] - targets[4]]
		buffer6 = state[ig - targets[0] - targets[3] - targets[4]]
		buffer7 = state[ig - targets[3] - targets[4]]
		buffer8 = state[ig - targets[0] - targets[1] - targets[2] - targets[4]]
		buffer9 = state[ig - targets[1] - targets[2] - targets[4]]
		buffer10 = state[ig - targets[0] - targets[2] - targets[4]]
		buffer11 = state[ig - targets[2] - targets[4]]
		buffer12 = state[ig - targets[0] - targets[1] - targets[4]]
		buffer13 = state[ig - targets[1] - targets[4]]
		buffer14 = state[ig - targets[0] - targets[4]]
		buffer15 = state[ig - targets[4]]
		buffer16 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]]
		buffer17 = state[ig - targets[1] - targets[2] - targets[3]]
		buffer18 = state[ig - targets[0] - targets[2] - targets[3]]
		buffer19 = state[ig - targets[2] - targets[3]]
		buffer20 = state[ig - targets[0] - targets[1] - targets[3]]
		buffer21 = state[ig - targets[1] - targets[3]]
		buffer22 = state[ig - targets[0] - targets[3]]
		buffer23 = state[ig - targets[3]]
		buffer24 = state[ig - targets[0] - targets[1] - targets[2]]
		buffer25 = state[ig - targets[1] - targets[2]]
		buffer26 = state[ig - targets[0] - targets[2]]
		buffer27 = state[ig - targets[2]]
		buffer28 = state[ig - targets[0] - targets[1]]
		buffer29 = state[ig - targets[1]]
		buffer30 = state[ig - targets[0]]
		buffer31 = state[ig]
		for i in range(32):
			t = ig - multitarget_index(31 - i, targets)
			state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15 + gate[i, 16] * buffer16 + gate[i, 17] * buffer17 + gate[i, 18] * buffer18 + gate[i, 19] * buffer19 + gate[i, 20] * buffer20 + gate[i, 21] * buffer21 + gate[i, 22] * buffer22 + gate[i, 23] * buffer23 + gate[i, 24] * buffer24 + gate[i, 25] * buffer25 + gate[i, 26] * buffer26 + gate[i, 27] * buffer27 + gate[i, 28] * buffer28 + gate[i, 29] * buffer29 + gate[i, 30] * buffer30 + gate[i, 31] * buffer31
	return state


@njit(parallel=True, cache=True)
def apply_six_qubit_gate_kernel(state, gate, qubits, nstates, targets):
	for g in prange(nstates):  # pylint: disable=not-an-iterable
		ig = multicontrol_index(g, qubits)
		buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer1 = state[ig - targets[1] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer2 = state[ig - targets[0] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer3 = state[ig - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer4 = state[ig - targets[0] - targets[1] - targets[3] - targets[4] - targets[5]]
		buffer5 = state[ig - targets[1] - targets[3] - targets[4] - targets[5]]
		buffer6 = state[ig - targets[0] - targets[3] - targets[4] - targets[5]]
		buffer7 = state[ig - targets[3] - targets[4] - targets[5]]
		buffer8 = state[ig - targets[0] - targets[1] - targets[2] - targets[4] - targets[5]]
		buffer9 = state[ig - targets[1] - targets[2] - targets[4] - targets[5]]
		buffer10 = state[ig - targets[0] - targets[2] - targets[4] - targets[5]]
		buffer11 = state[ig - targets[2] - targets[4] - targets[5]]
		buffer12 = state[ig - targets[0] - targets[1] - targets[4] - targets[5]]
		buffer13 = state[ig - targets[1] - targets[4] - targets[5]]
		buffer14 = state[ig - targets[0] - targets[4] - targets[5]]
		buffer15 = state[ig - targets[4] - targets[5]]
		buffer16 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[5]]
		buffer17 = state[ig - targets[1] - targets[2] - targets[3] - targets[5]]
		buffer18 = state[ig - targets[0] - targets[2] - targets[3] - targets[5]]
		buffer19 = state[ig - targets[2] - targets[3] - targets[5]]
		buffer20 = state[ig - targets[0] - targets[1] - targets[3] - targets[5]]
		buffer21 = state[ig - targets[1] - targets[3] - targets[5]]
		buffer22 = state[ig - targets[0] - targets[3] - targets[5]]
		buffer23 = state[ig - targets[3] - targets[5]]
		buffer24 = state[ig - targets[0] - targets[1] - targets[2] - targets[5]]
		buffer25 = state[ig - targets[1] - targets[2] - targets[5]]
		buffer26 = state[ig - targets[0] - targets[2] - targets[5]]
		buffer27 = state[ig - targets[2] - targets[5]]
		buffer28 = state[ig - targets[0] - targets[1] - targets[5]]
		buffer29 = state[ig - targets[1] - targets[5]]
		buffer30 = state[ig - targets[0] - targets[5]]
		buffer31 = state[ig - targets[5]]
		buffer32 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer33 = state[ig - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer34 = state[ig - targets[0] - targets[2] - targets[3] - targets[4]]
		buffer35 = state[ig - targets[2] - targets[3] - targets[4]]
		buffer36 = state[ig - targets[0] - targets[1] - targets[3] - targets[4]]
		buffer37 = state[ig - targets[1] - targets[3] - targets[4]]
		buffer38 = state[ig - targets[0] - targets[3] - targets[4]]
		buffer39 = state[ig - targets[3] - targets[4]]
		buffer40 = state[ig - targets[0] - targets[1] - targets[2] - targets[4]]
		buffer41 = state[ig - targets[1] - targets[2] - targets[4]]
		buffer42 = state[ig - targets[0] - targets[2] - targets[4]]
		buffer43 = state[ig - targets[2] - targets[4]]
		buffer44 = state[ig - targets[0] - targets[1] - targets[4]]
		buffer45 = state[ig - targets[1] - targets[4]]
		buffer46 = state[ig - targets[0] - targets[4]]
		buffer47 = state[ig - targets[4]]
		buffer48 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]]
		buffer49 = state[ig - targets[1] - targets[2] - targets[3]]
		buffer50 = state[ig - targets[0] - targets[2] - targets[3]]
		buffer51 = state[ig - targets[2] - targets[3]]
		buffer52 = state[ig - targets[0] - targets[1] - targets[3]]
		buffer53 = state[ig - targets[1] - targets[3]]
		buffer54 = state[ig - targets[0] - targets[3]]
		buffer55 = state[ig - targets[3]]
		buffer56 = state[ig - targets[0] - targets[1] - targets[2]]
		buffer57 = state[ig - targets[1] - targets[2]]
		buffer58 = state[ig - targets[0] - targets[2]]
		buffer59 = state[ig - targets[2]]
		buffer60 = state[ig - targets[0] - targets[1]]
		buffer61 = state[ig - targets[1]]
		buffer62 = state[ig - targets[0]]
		buffer63 = state[ig]
		for i in range(64):
			t = ig - multitarget_index(63 - i, targets)
			state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15 + gate[i, 16] * buffer16 + gate[i, 17] * buffer17 + gate[i, 18] * buffer18 + gate[i, 19] * buffer19 + gate[i, 20] * buffer20 + gate[i, 21] * buffer21 + gate[i, 22] * buffer22 + gate[i, 23] * buffer23 + gate[i, 24] * buffer24 + gate[i, 25] * buffer25 + gate[i, 26] * buffer26 + gate[i, 27] * buffer27 + gate[i, 28] * buffer28 + gate[i, 29] * buffer29 + gate[i, 30] * buffer30 + gate[i, 31] * buffer31 + gate[i, 32] * buffer32 + gate[i, 33] * buffer33 + gate[i, 34] * buffer34 + gate[i, 35] * buffer35 + gate[i, 36] * buffer36 + gate[i, 37] * buffer37 + gate[i, 38] * buffer38 + gate[i, 39] * buffer39 + gate[i, 40] * buffer40 + gate[i, 41] * buffer41 + gate[i, 42] * buffer42 + gate[i, 43] * buffer43 + gate[i, 44] * buffer44 + gate[i, 45] * buffer45 + gate[i, 46] * buffer46 + gate[i, 47] * buffer47 + gate[i, 48] * buffer48 + gate[i, 49] * buffer49 + gate[i, 50] * buffer50 + gate[i, 51] * buffer51 + gate[i, 52] * buffer52 + gate[i, 53] * buffer53 + gate[i, 54] * buffer54 + gate[i, 55] * buffer55 + gate[i, 56] * buffer56 + gate[i, 57] * buffer57 + gate[i, 58] * buffer58 + gate[i, 59] * buffer59 + gate[i, 60] * buffer60 + gate[i, 61] * buffer61 + gate[i, 62] * buffer62 + gate[i, 63] * buffer63
	return state


@njit(parallel=True, cache=True)
def apply_seven_qubit_gate_kernel(state, gate, qubits, nstates, targets):
	for g in prange(nstates):  # pylint: disable=not-an-iterable
		ig = multicontrol_index(g, qubits)
		buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer1 = state[ig - targets[1] - targets[2] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer2 = state[ig - targets[0] - targets[2] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer3 = state[ig - targets[2] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer4 = state[ig - targets[0] - targets[1] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer5 = state[ig - targets[1] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer6 = state[ig - targets[0] - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer7 = state[ig - targets[3] - targets[4] - targets[5] - targets[6]]
		buffer8 = state[ig - targets[0] - targets[1] - targets[2] - targets[4] - targets[5] - targets[6]]
		buffer9 = state[ig - targets[1] - targets[2] - targets[4] - targets[5] - targets[6]]
		buffer10 = state[ig - targets[0] - targets[2] - targets[4] - targets[5] - targets[6]]
		buffer11 = state[ig - targets[2] - targets[4] - targets[5] - targets[6]]
		buffer12 = state[ig - targets[0] - targets[1] - targets[4] - targets[5] - targets[6]]
		buffer13 = state[ig - targets[1] - targets[4] - targets[5] - targets[6]]
		buffer14 = state[ig - targets[0] - targets[4] - targets[5] - targets[6]]
		buffer15 = state[ig - targets[4] - targets[5] - targets[6]]
		buffer16 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[5] - targets[6]]
		buffer17 = state[ig - targets[1] - targets[2] - targets[3] - targets[5] - targets[6]]
		buffer18 = state[ig - targets[0] - targets[2] - targets[3] - targets[5] - targets[6]]
		buffer19 = state[ig - targets[2] - targets[3] - targets[5] - targets[6]]
		buffer20 = state[ig - targets[0] - targets[1] - targets[3] - targets[5] - targets[6]]
		buffer21 = state[ig - targets[1] - targets[3] - targets[5] - targets[6]]
		buffer22 = state[ig - targets[0] - targets[3] - targets[5] - targets[6]]
		buffer23 = state[ig - targets[3] - targets[5] - targets[6]]
		buffer24 = state[ig - targets[0] - targets[1] - targets[2] - targets[5] - targets[6]]
		buffer25 = state[ig - targets[1] - targets[2] - targets[5] - targets[6]]
		buffer26 = state[ig - targets[0] - targets[2] - targets[5] - targets[6]]
		buffer27 = state[ig - targets[2] - targets[5] - targets[6]]
		buffer28 = state[ig - targets[0] - targets[1] - targets[5] - targets[6]]
		buffer29 = state[ig - targets[1] - targets[5] - targets[6]]
		buffer30 = state[ig - targets[0] - targets[5] - targets[6]]
		buffer31 = state[ig - targets[5] - targets[6]]
		buffer32 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4] - targets[6]]
		buffer33 = state[ig - targets[1] - targets[2] - targets[3] - targets[4] - targets[6]]
		buffer34 = state[ig - targets[0] - targets[2] - targets[3] - targets[4] - targets[6]]
		buffer35 = state[ig - targets[2] - targets[3] - targets[4] - targets[6]]
		buffer36 = state[ig - targets[0] - targets[1] - targets[3] - targets[4] - targets[6]]
		buffer37 = state[ig - targets[1] - targets[3] - targets[4] - targets[6]]
		buffer38 = state[ig - targets[0] - targets[3] - targets[4] - targets[6]]
		buffer39 = state[ig - targets[3] - targets[4] - targets[6]]
		buffer40 = state[ig - targets[0] - targets[1] - targets[2] - targets[4] - targets[6]]
		buffer41 = state[ig - targets[1] - targets[2] - targets[4] - targets[6]]
		buffer42 = state[ig - targets[0] - targets[2] - targets[4] - targets[6]]
		buffer43 = state[ig - targets[2] - targets[4] - targets[6]]
		buffer44 = state[ig - targets[0] - targets[1] - targets[4] - targets[6]]
		buffer45 = state[ig - targets[1] - targets[4] - targets[6]]
		buffer46 = state[ig - targets[0] - targets[4] - targets[6]]
		buffer47 = state[ig - targets[4] - targets[6]]
		buffer48 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[6]]
		buffer49 = state[ig - targets[1] - targets[2] - targets[3] - targets[6]]
		buffer50 = state[ig - targets[0] - targets[2] - targets[3] - targets[6]]
		buffer51 = state[ig - targets[2] - targets[3] - targets[6]]
		buffer52 = state[ig - targets[0] - targets[1] - targets[3] - targets[6]]
		buffer53 = state[ig - targets[1] - targets[3] - targets[6]]
		buffer54 = state[ig - targets[0] - targets[3] - targets[6]]
		buffer55 = state[ig - targets[3] - targets[6]]
		buffer56 = state[ig - targets[0] - targets[1] - targets[2] - targets[6]]
		buffer57 = state[ig - targets[1] - targets[2] - targets[6]]
		buffer58 = state[ig - targets[0] - targets[2] - targets[6]]
		buffer59 = state[ig - targets[2] - targets[6]]
		buffer60 = state[ig - targets[0] - targets[1] - targets[6]]
		buffer61 = state[ig - targets[1] - targets[6]]
		buffer62 = state[ig - targets[0] - targets[6]]
		buffer63 = state[ig - targets[6]]
		buffer64 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer65 = state[ig - targets[1] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer66 = state[ig - targets[0] - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer67 = state[ig - targets[2] - targets[3] - targets[4] - targets[5]]
		buffer68 = state[ig - targets[0] - targets[1] - targets[3] - targets[4] - targets[5]]
		buffer69 = state[ig - targets[1] - targets[3] - targets[4] - targets[5]]
		buffer70 = state[ig - targets[0] - targets[3] - targets[4] - targets[5]]
		buffer71 = state[ig - targets[3] - targets[4] - targets[5]]
		buffer72 = state[ig - targets[0] - targets[1] - targets[2] - targets[4] - targets[5]]
		buffer73 = state[ig - targets[1] - targets[2] - targets[4] - targets[5]]
		buffer74 = state[ig - targets[0] - targets[2] - targets[4] - targets[5]]
		buffer75 = state[ig - targets[2] - targets[4] - targets[5]]
		buffer76 = state[ig - targets[0] - targets[1] - targets[4] - targets[5]]
		buffer77 = state[ig - targets[1] - targets[4] - targets[5]]
		buffer78 = state[ig - targets[0] - targets[4] - targets[5]]
		buffer79 = state[ig - targets[4] - targets[5]]
		buffer80 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[5]]
		buffer81 = state[ig - targets[1] - targets[2] - targets[3] - targets[5]]
		buffer82 = state[ig - targets[0] - targets[2] - targets[3] - targets[5]]
		buffer83 = state[ig - targets[2] - targets[3] - targets[5]]
		buffer84 = state[ig - targets[0] - targets[1] - targets[3] - targets[5]]
		buffer85 = state[ig - targets[1] - targets[3] - targets[5]]
		buffer86 = state[ig - targets[0] - targets[3] - targets[5]]
		buffer87 = state[ig - targets[3] - targets[5]]
		buffer88 = state[ig - targets[0] - targets[1] - targets[2] - targets[5]]
		buffer89 = state[ig - targets[1] - targets[2] - targets[5]]
		buffer90 = state[ig - targets[0] - targets[2] - targets[5]]
		buffer91 = state[ig - targets[2] - targets[5]]
		buffer92 = state[ig - targets[0] - targets[1] - targets[5]]
		buffer93 = state[ig - targets[1] - targets[5]]
		buffer94 = state[ig - targets[0] - targets[5]]
		buffer95 = state[ig - targets[5]]
		buffer96 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer97 = state[ig - targets[1] - targets[2] - targets[3] - targets[4]]
		buffer98 = state[ig - targets[0] - targets[2] - targets[3] - targets[4]]
		buffer99 = state[ig - targets[2] - targets[3] - targets[4]]
		buffer100 = state[ig - targets[0] - targets[1] - targets[3] - targets[4]]
		buffer101 = state[ig - targets[1] - targets[3] - targets[4]]
		buffer102 = state[ig - targets[0] - targets[3] - targets[4]]
		buffer103 = state[ig - targets[3] - targets[4]]
		buffer104 = state[ig - targets[0] - targets[1] - targets[2] - targets[4]]
		buffer105 = state[ig - targets[1] - targets[2] - targets[4]]
		buffer106 = state[ig - targets[0] - targets[2] - targets[4]]
		buffer107 = state[ig - targets[2] - targets[4]]
		buffer108 = state[ig - targets[0] - targets[1] - targets[4]]
		buffer109 = state[ig - targets[1] - targets[4]]
		buffer110 = state[ig - targets[0] - targets[4]]
		buffer111 = state[ig - targets[4]]
		buffer112 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]]
		buffer113 = state[ig - targets[1] - targets[2] - targets[3]]
		buffer114 = state[ig - targets[0] - targets[2] - targets[3]]
		buffer115 = state[ig - targets[2] - targets[3]]
		buffer116 = state[ig - targets[0] - targets[1] - targets[3]]
		buffer117 = state[ig - targets[1] - targets[3]]
		buffer118 = state[ig - targets[0] - targets[3]]
		buffer119 = state[ig - targets[3]]
		buffer120 = state[ig - targets[0] - targets[1] - targets[2]]
		buffer121 = state[ig - targets[1] - targets[2]]
		buffer122 = state[ig - targets[0] - targets[2]]
		buffer123 = state[ig - targets[2]]
		buffer124 = state[ig - targets[0] - targets[1]]
		buffer125 = state[ig - targets[1]]
		buffer126 = state[ig - targets[0]]
		buffer127 = state[ig]
		for i in range(128):
			t = ig - multitarget_index(127 - i, targets)
			state[t] = gate[i, 0] * buffer0 + gate[i, 1] * buffer1 + gate[i, 2] * buffer2 + gate[i, 3] * buffer3 + gate[i, 4] * buffer4 + gate[i, 5] * buffer5 + gate[i, 6] * buffer6 + gate[i, 7] * buffer7 + gate[i, 8] * buffer8 + gate[i, 9] * buffer9 + gate[i, 10] * buffer10 + gate[i, 11] * buffer11 + gate[i, 12] * buffer12 + gate[i, 13] * buffer13 + gate[i, 14] * buffer14 + gate[i, 15] * buffer15 + gate[i, 16] * buffer16 + gate[i, 17] * buffer17 + gate[i, 18] * buffer18 + gate[i, 19] * buffer19 + gate[i, 20] * buffer20 + gate[i, 21] * buffer21 + gate[i, 22] * buffer22 + gate[i, 23] * buffer23 + gate[i, 24] * buffer24 + gate[i, 25] * buffer25 + gate[i, 26] * buffer26 + gate[i, 27] * buffer27 + gate[i, 28] * buffer28 + gate[i, 29] * buffer29 + gate[i, 30] * buffer30 + gate[i, 31] * buffer31 + gate[i, 32] * buffer32 + gate[i, 33] * buffer33 + gate[i, 34] * buffer34 + gate[i, 35] * buffer35 + gate[i, 36] * buffer36 + gate[i, 37] * buffer37 + gate[i, 38] * buffer38 + gate[i, 39] * buffer39 + gate[i, 40] * buffer40 + gate[i, 41] * buffer41 + gate[i, 42] * buffer42 + gate[i, 43] * buffer43 + gate[i, 44] * buffer44 + gate[i, 45] * buffer45 + gate[i, 46] * buffer46 + gate[i, 47] * buffer47 + gate[i, 48] * buffer48 + gate[i, 49] * buffer49 + gate[i, 50] * buffer50 + gate[i, 51] * buffer51 + gate[i, 52] * buffer52 + gate[i, 53] * buffer53 + gate[i, 54] * buffer54 + gate[i, 55] * buffer55 + gate[i, 56] * buffer56 + gate[i, 57] * buffer57 + gate[i, 58] * buffer58 + gate[i, 59] * buffer59 + gate[i, 60] * buffer60 + gate[i, 61] * buffer61 + gate[i, 62] * buffer62 + gate[i, 63] * buffer63 + gate[i, 64] * buffer64 + gate[i, 65] * buffer65 + gate[i, 66] * buffer66 + gate[i, 67] * buffer67 + gate[i, 68] * buffer68 + gate[i, 69] * buffer69 + gate[i, 70] * buffer70 + gate[i, 71] * buffer71 + gate[i, 72] * buffer72 + gate[i, 73] * buffer73 + gate[i, 74] * buffer74 + gate[i, 75] * buffer75 + gate[i, 76] * buffer76 + gate[i, 77] * buffer77 + gate[i, 78] * buffer78 + gate[i, 79] * buffer79 + gate[i, 80] * buffer80 + gate[i, 81] * buffer81 + gate[i, 82] * buffer82 + gate[i, 83] * buffer83 + gate[i, 84] * buffer84 + gate[i, 85] * buffer85 + gate[i, 86] * buffer86 + gate[i, 87] * buffer87 + gate[i, 88] * buffer88 + gate[i, 89] * buffer89 + gate[i, 90] * buffer90 + gate[i, 91] * buffer91 + gate[i, 92] * buffer92 + gate[i, 93] * buffer93 + gate[i, 94] * buffer94 + gate[i, 95] * buffer95 + gate[i, 96] * buffer96 + gate[i, 97] * buffer97 + gate[i, 98] * buffer98 + gate[i, 99] * buffer99 + gate[i, 100] * buffer100 + gate[i, 101] * buffer101 + gate[i, 102] * buffer102 + gate[i, 103] * buffer103 + gate[i, 104] * buffer104 + gate[i, 105] * buffer105 + gate[i, 106] * buffer106 + gate[i, 107] * buffer107 + gate[i, 108] * buffer108 + gate[i, 109] * buffer109 + gate[i, 110] * buffer110 + gate[i, 111] * buffer111 + gate[i, 112] * buffer112 + gate[i, 113] * buffer113 + gate[i, 114] * buffer114 + gate[i, 115] * buffer115 + gate[i, 116] * buffer116 + gate[i, 117] * buffer117 + gate[i, 118] * buffer118 + gate[i, 119] * buffer119 + gate[i, 120] * buffer120 + gate[i, 121] * buffer121 + gate[i, 122] * buffer122 + gate[i, 123] * buffer123 + gate[i, 124] * buffer124 + gate[i, 125] * buffer125 + gate[i, 126] * buffer126 + gate[i, 127] * buffer127
	return state
