class NumbaBackend:

    def __init__(self):
        from qibojit.custom_operators import gates, ops
        self.gates = gates
        self.ops = ops

    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        nstates = 1 << (nqubits - ncontrols - 1)
        kernel = getattr(self.gates, f"{kernel}_kernel")
        if ncontrols:
            return self.gates.one_qubit_multicontrol(state, gate, kernel, qubits, nstates, m)
        return self.gates.one_qubit_nocontrol(state, gate, kernel, nstates, m)

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None):
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
        kernel = getattr(self.gates, f"{kernel}_kernel")
        if ncontrols:
            return self.gates.two_qubit_multicontrol(state, gate, kernel, qubits, nstates, m1, m2, swap_targets)
        return self.gates.two_qubit_nocontrol(state, gate, kernel, nstates, m1, m2, swap_targets)

    def initial_state(self, nqubits, dtype, is_matrix=False):
        return self.ops.initial_state(nqubits, dtype, is_matrix)

    def collapse_state(self, state, qubits, result, nqubits, normalize):
        return self.ops.collapse_state(state, qubits, result, nqubits, normalize)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        return self.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234)


class CupyBackend:

    DEFAULT_BLOCK_SIZE = 1024

    def __init__(self):
        import cupy as cp
        self.cp = cp
        gates_dir = os.path.dirname(os.path.realpath(__file__))
        gates_dir = os.path.join(module_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            self.gates = cp.RawModule(code=r"{}".format(file.read()))

    def calculate_blocks(self, nstates):
        block_size = self.DEFAULT_BLOCK_SIZE
        nblocks = (nstates + block_size - 1) // block_size
        if nstates < block_size:
            nblocks = 1
            block_size = nstates
        return nblocks, block_size

    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)

        if gate is None:
            args = (state, tk, m)
        else:
            args = (state, tk, m, gate)

        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel")
            args += (qubits, ncontrols)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 2 if qubits is not None else 0
        t1, t2 = max(target1, target2), min(target1, target2)
        m1, m2 = nqubits - t1 - 1, nqubits - t2 - 1
        tk1, tk2 = 1 << m1, 1 << m2
        nstates = 1 << (nqubits - 2 - ncontrols)

        if gate is None:
            args = (state, tk1, tk2, m1, m2)
        else:
            args = (state, tk1, tk2, m1, m2, gate)

        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel")
            args += (qubits, ncontrols)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def initial_state(self, nqubits, dtype, is_matrix=False):
        raise NotImplementedError

    def collapse_state(self, state, qubits, result, nqubits, normalize):
        raise NotImplementedError

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        raise NotImplementedError


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
        return self.backend.initial_state(nqubits, dtype, is_matrix)

    def collapse_state(self, state, qubits, result, nqubits, normalize):
        return self.backend.collapse_state(state, qubits, result, nqubits, normalize)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        return self.backend.measure_frequencies(frequencies, probs, nshots, nqubits, seed=1234)


backend = Backend()
