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


def generate_multiqubit_gate_kernel(ntargets):
    n = 2 ** ntargets
    yield f"def apply_multi{ntargets}_gate_kernel(state, gate, qubits, nstates, indices):"
    yield f"\tfor g in prange(nstates):"
    yield f"\t\tig = multicontrol_index(g, qubits) - indices[{n - 1}]"
    for i in range(n):
        yield f"\t\tbuffer{i} = state[ig + indices[{i}]]"
        new_state = [f"gate[{i}, {j}] * buffer{j}" for j in range(min(i + 1, n))]
        new_state.extend(f"gate[{i}, {j}] * state[ig + indices[{j}]]" for j in range(i + 1, n))
        new_state = " + ".join(new_state)
        yield f"\t\tstate[ig + indices[{i}]] = {new_state}"
    yield f"\treturn state"


def create_multiqubit_kernel(ntargets):
    code = "\n".join(generate_multiqubit_gate_kernel(ntargets))
    code = compile(code, "<string>", "exec")
    kernel = FunctionType(code.co_consts[0], globals())
    return njit(parallel=True)(kernel)


@njit(parallel=True, cache=True)
def apply_three_qubit_gate_kernel(state, gate, qubits, nstates, indices):
    for g in prange(nstates):
        ig = multicontrol_index(g, qubits) - indices[7]
        buffer0 = state[ig + indices[0]]
        buffer1 = state[ig + indices[1]]
        buffer2 = state[ig + indices[2]]
        buffer3 = state[ig + indices[3]]
        buffer4 = state[ig + indices[4]]
        buffer5 = state[ig + indices[5]]
        buffer6 = state[ig + indices[6]]
        buffer7 = state[ig + indices[7]]
        state[ig + indices[0]] = gate[0, 0] * buffer0 + gate[0, 1] * state[ig + indices[1]] + gate[0, 2] * state[ig + indices[2]] + gate[0, 3] * state[ig + indices[3]] + gate[0, 4] * state[ig + indices[4]] + gate[0, 5] * state[ig + indices[5]] + gate[0, 6] * state[ig + indices[6]] + gate[0, 7] * state[ig + indices[7]]
        state[ig + indices[1]] = gate[1, 0] * buffer0 + gate[1, 1] * buffer1 + gate[1, 2] * state[ig + indices[2]] + gate[1, 3] * state[ig + indices[3]] + gate[1, 4] * state[ig + indices[4]] + gate[1, 5] * state[ig + indices[5]] + gate[1, 6] * state[ig + indices[6]] + gate[1, 7] * state[ig + indices[7]]
        state[ig + indices[2]] = gate[2, 0] * buffer0 + gate[2, 1] * buffer1 + gate[2, 2] * buffer2 + gate[2, 3] * state[ig + indices[3]] + gate[2, 4] * state[ig + indices[4]] + gate[2, 5] * state[ig + indices[5]] + gate[2, 6] * state[ig + indices[6]] + gate[2, 7] * state[ig + indices[7]]
        state[ig + indices[3]] = gate[3, 0] * buffer0 + gate[3, 1] * buffer1 + gate[3, 2] * buffer2 + gate[3, 3] * buffer3 + gate[3, 4] * state[ig + indices[4]] + gate[3, 5] * state[ig + indices[5]] + gate[3, 6] * state[ig + indices[6]] + gate[3, 7] * state[ig + indices[7]]
        state[ig + indices[4]] = gate[4, 0] * buffer0 + gate[4, 1] * buffer1 + gate[4, 2] * buffer2 + gate[4, 3] * buffer3 + gate[4, 4] * buffer4 + gate[4, 5] * state[ig + indices[5]] + gate[4, 6] * state[ig + indices[6]] + gate[4, 7] * state[ig + indices[7]]
        state[ig + indices[5]] = gate[5, 0] * buffer0 + gate[5, 1] * buffer1 + gate[5, 2] * buffer2 + gate[5, 3] * buffer3 + gate[5, 4] * buffer4 + gate[5, 5] * buffer5 + gate[5, 6] * state[ig + indices[6]] + gate[5, 7] * state[ig + indices[7]]
        state[ig + indices[6]] = gate[6, 0] * buffer0 + gate[6, 1] * buffer1 + gate[6, 2] * buffer2 + gate[6, 3] * buffer3 + gate[6, 4] * buffer4 + gate[6, 5] * buffer5 + gate[6, 6] * buffer6 + gate[6, 7] * state[ig + indices[7]]
        state[ig + indices[7]] = gate[7, 0] * buffer0 + gate[7, 1] * buffer1 + gate[7, 2] * buffer2 + gate[7, 3] * buffer3 + gate[7, 4] * buffer4 + gate[7, 5] * buffer5 + gate[7, 6] * buffer6 + gate[7, 7] * buffer7
    return state


@njit(parallel=True, cache=True)
def apply_four_qubit_gate_kernel(state, gate, qubits, nstates, indices):
    for g in prange(nstates):
        ig = multicontrol_index(g, qubits) - indices[15]
        buffer0 = state[ig + indices[0]]
        buffer1 = state[ig + indices[1]]
        buffer2 = state[ig + indices[2]]
        buffer3 = state[ig + indices[3]]
        buffer4 = state[ig + indices[4]]
        buffer5 = state[ig + indices[5]]
        buffer6 = state[ig + indices[6]]
        buffer7 = state[ig + indices[7]]
        buffer8 = state[ig + indices[8]]
        buffer9 = state[ig + indices[9]]
        buffer10 = state[ig + indices[10]]
        buffer11 = state[ig + indices[11]]
        buffer12 = state[ig + indices[12]]
        buffer13 = state[ig + indices[13]]
        buffer14 = state[ig + indices[14]]
        buffer15 = state[ig + indices[15]]

        state[ig + indices[0]] = (gate[0, 0] * buffer0 + gate[0, 1] * state[ig + indices[1]] + gate[0, 2] * state[ig + indices[2]] + gate[0, 3] * state[ig + indices[3]] + gate[0, 4] * state[ig + indices[4]] + gate[0, 5] * state[ig + indices[5]] + gate[0, 6] * state[ig + indices[6]] + gate[0, 7] * state[ig + indices[7]] + gate[0, 8] * state[ig + indices[8]] + gate[0, 9] * state[ig + indices[9]] + gate[0, 10] * state[ig + indices[10]] + gate[0, 11] * state[ig + indices[11]] + gate[0, 12] * state[ig + indices[12]] + gate[0, 13] * state[ig + indices[13]] + gate[0, 14] * state[ig + indices[14]] + gate[0, 15] * state[ig + indices[15]])
        state[ig + indices[1]] = (gate[1, 0] * buffer0 + gate[1, 1] * buffer1 + gate[1, 2] * state[ig + indices[2]] + gate[1, 3] * state[ig + indices[3]] + gate[1, 4] * state[ig + indices[4]] + gate[1, 5] * state[ig + indices[5]] + gate[1, 6] * state[ig + indices[6]] + gate[1, 7] * state[ig + indices[7]] + gate[1, 8] * state[ig + indices[8]] + gate[1, 9] * state[ig + indices[9]] + gate[1, 10] * state[ig + indices[10]] + gate[1, 11] * state[ig + indices[11]] + gate[1, 12] * state[ig + indices[12]] + gate[1, 13] * state[ig + indices[13]] + gate[1, 14] * state[ig + indices[14]] + gate[1, 15] * state[ig + indices[15]])
        state[ig + indices[2]] = gate[2, 0] * buffer0 + gate[2, 1] * buffer1 + gate[2, 2] * buffer2 + gate[2, 3] * state[ig + indices[3]] + gate[2, 4] * state[ig + indices[4]] + gate[2, 5] * state[ig + indices[5]] + gate[2, 6] * state[ig + indices[6]] + gate[2, 7] * state[ig + indices[7]] + gate[2, 8] * state[ig + indices[8]] + gate[2, 9] * state[ig + indices[9]] + gate[2, 10] * state[ig + indices[10]] + gate[2, 11] * state[ig + indices[11]] + gate[2, 12] * state[ig + indices[12]] + gate[2, 13] * state[ig + indices[13]] + gate[2, 14] * state[ig + indices[14]] + gate[2, 15] * state[ig + indices[15]]
        state[ig + indices[3]] = gate[3, 0] * buffer0 + gate[3, 1] * buffer1 + gate[3, 2] * buffer2 + gate[3, 3] * buffer3 + gate[3, 4] * state[ig + indices[4]] + gate[3, 5] * state[ig + indices[5]] + gate[3, 6] * state[ig + indices[6]] + gate[3, 7] * state[ig + indices[7]] + gate[3, 8] * state[ig + indices[8]] + gate[3, 9] * state[ig + indices[9]] + gate[3, 10] * state[ig + indices[10]] + gate[3, 11] * state[ig + indices[11]] + gate[3, 12] * state[ig + indices[12]] + gate[3, 13] * state[ig + indices[13]] + gate[3, 14] * state[ig + indices[14]] + gate[3, 15] * state[ig + indices[15]]
        state[ig + indices[4]] = gate[4, 0] * buffer0 + gate[4, 1] * buffer1 + gate[4, 2] * buffer2 + gate[4, 3] * buffer3 + gate[4, 4] * buffer4 + gate[4, 5] * state[ig + indices[5]] + gate[4, 6] * state[ig + indices[6]] + gate[4, 7] * state[ig + indices[7]] + gate[4, 8] * state[ig + indices[8]] + gate[4, 9] * state[ig + indices[9]] + gate[4, 10] * state[ig + indices[10]] + gate[4, 11] * state[ig + indices[11]] + gate[4, 12] * state[ig + indices[12]] + gate[4, 13] * state[ig + indices[13]] + gate[4, 14] * state[ig + indices[14]] + gate[4, 15] * state[ig + indices[15]]
        state[ig + indices[5]] = gate[5, 0] * buffer0 + gate[5, 1] * buffer1 + gate[5, 2] * buffer2 + gate[5, 3] * buffer3 + gate[5, 4] * buffer4 + gate[5, 5] * buffer5 + gate[5, 6] * state[ig + indices[6]] + gate[5, 7] * state[ig + indices[7]] + gate[5, 8] * state[ig + indices[8]] + gate[5, 9] * state[ig + indices[9]] + gate[5, 10] * state[ig + indices[10]] + gate[5, 11] * state[ig + indices[11]] + gate[5, 12] * state[ig + indices[12]] + gate[5, 13] * state[ig + indices[13]] + gate[5, 14] * state[ig + indices[14]] + gate[5, 15] * state[ig + indices[15]]
        state[ig + indices[6]] = gate[6, 0] * buffer0 + gate[6, 1] * buffer1 + gate[6, 2] * buffer2 + gate[6, 3] * buffer3 + gate[6, 4] * buffer4 + gate[6, 5] * buffer5 + gate[6, 6] * buffer6 + gate[6, 7] * state[ig + indices[7]] + gate[6, 8] * state[ig + indices[8]] + gate[6, 9] * state[ig + indices[9]] + gate[6, 10] * state[ig + indices[10]] + gate[6, 11] * state[ig + indices[11]] + gate[6, 12] * state[ig + indices[12]] + gate[6, 13] * state[ig + indices[13]] + gate[6, 14] * state[ig + indices[14]] + gate[6, 15] * state[ig + indices[15]]
        state[ig + indices[7]] = gate[7, 0] * buffer0 + gate[7, 1] * buffer1 + gate[7, 2] * buffer2 + gate[7, 3] * buffer3 + gate[7, 4] * buffer4 + gate[7, 5] * buffer5 + gate[7, 6] * buffer6 + gate[7, 7] * buffer7 + gate[7, 8] * state[ig + indices[8]] + gate[7, 9] * state[ig + indices[9]] + gate[7, 10] * state[ig + indices[10]] + gate[7, 11] * state[ig + indices[11]] + gate[7, 12] * state[ig + indices[12]] + gate[7, 13] * state[ig + indices[13]] + gate[7, 14] * state[ig + indices[14]] + gate[7, 15] * state[ig + indices[15]]
        state[ig + indices[8]] = gate[8, 0] * buffer0 + gate[8, 1] * buffer1 + gate[8, 2] * buffer2 + gate[8, 3] * buffer3 + gate[8, 4] * buffer4 + gate[8, 5] * buffer5 + gate[8, 6] * buffer6 + gate[8, 7] * buffer7 + gate[8, 8] * buffer8 + gate[8, 9] * state[ig + indices[9]] + gate[8, 10] * state[ig + indices[10]] + gate[8, 11] * state[ig + indices[11]] + gate[8, 12] * state[ig + indices[12]] + gate[8, 13] * state[ig + indices[13]] + gate[8, 14] * state[ig + indices[14]] + gate[8, 15] * state[ig + indices[15]]
        state[ig + indices[9]] = gate[9, 0] * buffer0 + gate[9, 1] * buffer1 + gate[9, 2] * buffer2 + gate[9, 3] * buffer3 + gate[9, 4] * buffer4 + gate[9, 5] * buffer5 + gate[9, 6] * buffer6 + gate[9, 7] * buffer7 + gate[9, 8] * buffer8 + gate[9, 9] * buffer9 + gate[9, 10] * state[ig + indices[10]] + gate[9, 11] * state[ig + indices[11]] + gate[9, 12] * state[ig + indices[12]] + gate[9, 13] * state[ig + indices[13]] + gate[9, 14] * state[ig + indices[14]] + gate[9, 15] * state[ig + indices[15]]
        state[ig + indices[10]] = gate[10, 0] * buffer0 + gate[10, 1] * buffer1 + gate[10, 2] * buffer2 + gate[10, 3] * buffer3 + gate[10, 4] * buffer4 + gate[10, 5] * buffer5 + gate[10, 6] * buffer6 + gate[10, 7] * buffer7 + gate[10, 8] * buffer8 + gate[10, 9] * buffer9 + gate[10, 10] * buffer10 + gate[10, 11] * state[ig + indices[11]] + gate[10, 12] * state[ig + indices[12]] + gate[10, 13] * state[ig + indices[13]] + gate[10, 14] * state[ig + indices[14]] + gate[10, 15] * state[ig + indices[15]]
        state[ig + indices[11]] = gate[11, 0] * buffer0 + gate[11, 1] * buffer1 + gate[11, 2] * buffer2 + gate[11, 3] * buffer3 + gate[11, 4] * buffer4 + gate[11, 5] * buffer5 + gate[11, 6] * buffer6 + gate[11, 7] * buffer7 + gate[11, 8] * buffer8 + gate[11, 9] * buffer9 + gate[11, 10] * buffer10 + gate[11, 11] * buffer11 + gate[11, 12] * state[ig + indices[12]] + gate[11, 13] * state[ig + indices[13]] + gate[11, 14] * state[ig + indices[14]] + gate[11, 15] * state[ig + indices[15]]
        state[ig + indices[12]] = gate[12, 0] * buffer0 + gate[12, 1] * buffer1 + gate[12, 2] * buffer2 + gate[12, 3] * buffer3 + gate[12, 4] * buffer4 + gate[12, 5] * buffer5 + gate[12, 6] * buffer6 + gate[12, 7] * buffer7 + gate[12, 8] * buffer8 + gate[12, 9] * buffer9 + gate[12, 10] * buffer10 + gate[12, 11] * buffer11 + gate[12, 12] * buffer12 + gate[12, 13] * state[ig + indices[13]] + gate[12, 14] * state[ig + indices[14]] + gate[12, 15] * state[ig + indices[15]]
        state[ig + indices[13]] = gate[13, 0] * buffer0 + gate[13, 1] * buffer1 + gate[13, 2] * buffer2 + gate[13, 3] * buffer3 + gate[13, 4] * buffer4 + gate[13, 5] * buffer5 + gate[13, 6] * buffer6 + gate[13, 7] * buffer7 + gate[13, 8] * buffer8 + gate[13, 9] * buffer9 + gate[13, 10] * buffer10 + gate[13, 11] * buffer11 + gate[13, 12] * buffer12 + gate[13, 13] * buffer13 + gate[13, 14] * state[ig + indices[14]] + gate[13, 15] * state[ig + indices[15]]
        state[ig + indices[14]] = gate[14, 0] * buffer0 + gate[14, 1] * buffer1 + gate[14, 2] * buffer2 + gate[14, 3] * buffer3 + gate[14, 4] * buffer4 + gate[14, 5] * buffer5 + gate[14, 6] * buffer6 + gate[14, 7] * buffer7 + gate[14, 8] * buffer8 + gate[14, 9] * buffer9 + gate[14, 10] * buffer10 + gate[14, 11] * buffer11 + gate[14, 12] * buffer12 + gate[14, 13] * buffer13 + gate[14, 14] * buffer14 + gate[14, 15] * state[ig + indices[15]]
        state[ig + indices[15]] = gate[15, 0] * buffer0 + gate[15, 1] * buffer1 + gate[15, 2] * buffer2 + gate[15, 3] * buffer3 + gate[15, 4] * buffer4 + gate[15, 5] * buffer5 + gate[15, 6] * buffer6 + gate[15, 7] * buffer7 + gate[15, 8] * buffer8 + gate[15, 9] * buffer9 + gate[15, 10] * buffer10 + gate[15, 11] * buffer11 + gate[15, 12] * buffer12 + gate[15, 13] * buffer13 + gate[15, 14] * buffer14 + gate[15, 15] * buffer15
    return state
