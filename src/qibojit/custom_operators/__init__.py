from qibojit.custom_operators import gates
from qibojit.custom_operators.ops import initial_state
from qibojit.custom_operators.ops import collapse_state
from qibojit.custom_operators.ops import measure_frequencies


def one_qubit_base(state, nqubits, target, kernel, qubits=None, gate=None):
    ncontrols = len(qubits) - 1 if qubits is not None else 0
    m = nqubits - target - 1
    nstates = 1 << (nqubits - ncontrols - 1)
    if ncontrols:
        kernel = getattr(gates, "multicontrol_{}_kernel".format(kernel))
        return kernel(state, gate, qubits, nstates, m)
    kernel = getattr(gates, "{}_kernel".format(kernel))
    return kernel(state, gate, nstates, m)


def two_qubit_base(state, nqubits, target1, target2, kernel, qubits=None, gate=None):
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
        kernel = getattr(gates, "multicontrol_{}_kernel".format(kernel))
        return kernel(state, gate, qubits, nstates, m1, m2, swap_targets)
    kernel = getattr(gates, "{}_kernel".format(kernel))
    return kernel(state, gate, nstates, m1, m2, swap_targets)


def apply_gate(state, gate, nqubits, target, qubits=None):
    return one_qubit_base(state, nqubits, target, "apply_gate", qubits, gate)

def apply_x(state, nqubits, target, qubits=None):
    return one_qubit_base(state, nqubits, target, "apply_x", qubits)

def apply_y(state, nqubits, target, qubits=None):
    return one_qubit_base(state, nqubits, target, "apply_y", qubits)

def apply_z(state, nqubits, target, qubits=None):
    return one_qubit_base(state, nqubits, target, "apply_z", qubits)

def apply_z_pow(state, gate, nqubits, target, qubits=None):
    return one_qubit_base(state, nqubits, target, "apply_z_pow", qubits, gate)

def apply_two_qubit_gate(state, gate, nqubits, target1, target2, qubits=None):
    return two_qubit_base(state, nqubits, target1, target2, "apply_two_qubit_gate",
                          qubits, gate)

def apply_swap(state, nqubits, target1, target2, qubits=None):
    return two_qubit_base(state, nqubits, target1, target2, "apply_swap",
                          qubits)

def apply_fsim(state, gate, nqubits, target1, target2, qubits=None):
    return two_qubit_base(state, nqubits, target1, target2, "apply_fsim",
                          qubits, gate)
