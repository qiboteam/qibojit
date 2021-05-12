from qibojit.custom_operators.backends import NumbaBackend, CupyBackend


class Backend:

    available_backends = {"numba": NumbaBackend, "cupy": CupyBackend}

    def __init__(self):
        self.constructed_backends = {}
        self.active_backend = self.construct_backend("numba")

    def construct_backend(self, name):
        if name not in self.constructed_backends:
            if name in self.available_backends:
                self.constructed_backends[name] = self.available_backends.get(name)()
            else:
                raise KeyError
        return self.constructed_backends.get(name)

    def set_backend(self, name):
        self.active_backend = self.construct_backend(name)

    def __getattr__(self, x):
        return getattr(self.active_backend, x)


backend = Backend()

def apply_gate(state, gate, nqubits, target, qubits=None):
    return backend.one_qubit_base(state, nqubits, target, "apply_gate", qubits, gate)


def apply_x(state, nqubits, target, qubits=None):
    return backend.one_qubit_base(state, nqubits, target, "apply_x", qubits)


def apply_y(state, nqubits, target, qubits=None):
    return backend.one_qubit_base(state, nqubits, target, "apply_y", qubits)


def apply_z(state, nqubits, target, qubits=None):
    return backend.one_qubit_base(state, nqubits, target, "apply_z", qubits)


def apply_z_pow(state, gate, nqubits, target, qubits=None):
    return backend.one_qubit_base(state, nqubits, target, "apply_z_pow", qubits, gate)


def apply_two_qubit_gate(state, gate, nqubits, target1, target2, qubits=None):
    return backend.two_qubit_base(state, nqubits, target1, target2,
                                  "apply_two_qubit_gate", qubits, gate)


def apply_swap(state, nqubits, target1, target2, qubits=None):
    return backend.two_qubit_base(state, nqubits, target1, target2, "apply_swap", qubits)


def apply_fsim(state, gate, nqubits, target1, target2, qubits=None):
    return backend.two_qubit_base(state, nqubits, target1, target2,
                                  "apply_fsim", qubits, gate)


def initial_state(nqubits, dtype, is_matrix=False):
    return backend.ops.initial_state(nqubits, dtype, is_matrix)


def collapse_state(state, qubits, result, nqubits, normalize):
    return backend.ops.collapse_state(state, qubits, result, nqubits, normalize)


def measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234):
    return backend.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed)