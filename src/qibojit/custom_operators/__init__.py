from qibojit.custom_operators.backends import NumbaBackend, CupyBackend


class Backend:

    available_backends = {"numba": NumbaBackend, "cupy": CupyBackend}

    def __init__(self):
        self.constructed_backends = {}
        self.set_backend("numba")

    def construct_backend(self, name):
        if name not in self.constructed_backends:
            if name in self.available_backends:
                self.constructed_backends[name] = self.available_backends.get(name)()
            else:
                raise KeyError
        return self.constructed_backends.get(name)

    def set_backend(self, name):
        self.backend = self.construct_backend(name)

    def apply_gate(self, state, gate, nqubits, target, qubits=None):
        return self.backend.one_qubit_base(state, nqubits, target, "apply_gate", qubits, gate)

    def apply_x(self, state, nqubits, target, qubits=None):
        return self.backend.one_qubit_base(state, nqubits, target, "apply_x", qubits)

    def apply_y(self, state, nqubits, target, qubits=None):
        return self.backend.one_qubit_base(state, nqubits, target, "apply_y", qubits)

    def apply_z(self, state, nqubits, target, qubits=None):
        return self.backend.one_qubit_base(state, nqubits, target, "apply_z", qubits)

    def apply_z_pow(self, state, gate, nqubits, target, qubits=None):
        return self.backend.one_qubit_base(state, nqubits, target, "apply_z_pow", qubits, gate)

    def apply_two_qubit_gate(self, state, gate, nqubits, target1, target2, qubits=None):
        return self.backend.two_qubit_base(state, nqubits, target1, target2,
                                           "apply_two_qubit_gate",
                                           qubits, gate)

    def apply_swap(self, state, nqubits, target1, target2, qubits=None):
        return self.backend.two_qubit_base(state, nqubits, target1, target2,
                                           "apply_swap", qubits)

    def apply_fsim(self, state, gate, nqubits, target1, target2, qubits=None):
        return self.backend.two_qubit_base(state, nqubits, target1, target2,
                                           "apply_fsim", qubits, gate)

    def initial_state(self, nqubits, dtype, is_matrix=False):
        return self.backend.ops.initial_state(nqubits, dtype, is_matrix)

    def collapse_state(self, state, qubits, result, nqubits, normalize):
        return self.backend.ops.collapse_state(state, qubits, result, nqubits, normalize)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        return self.backend.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed)


backend = Backend()
apply_gate = backend.apply_gate
apply_x = backend.apply_x
apply_y = backend.apply_y
apply_z = backend.apply_z
apply_z_pow = backend.apply_z_pow
apply_two_qubit_gate = backend.apply_two_qubit_gate
apply_swap = backend.apply_swap
apply_fsim = backend.apply_fsim
initial_state = backend.initial_state
collapse_state = backend.collapse_state
measure_frequencies = backend.measure_frequencies
