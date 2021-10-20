class AbstractBackend:

    def __init__(self):
        import numpy as np
        self.np = np
        self.op = None

    def cast(self, x, dtype=None):
        return self.op.cast(x, dtype=dtype)

    def to_numpy(self, x):
        return self.op.to_numpy(x)

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = [nqubits - q - 1 for q in targets]
        qubits.extend(nqubits - q - 1 for q in controls)
        return self.op.cast(sorted(qubits),dtype=self.np.int32)

    def apply_gate_args(self, state, nqubits, controls=[]):
        gate = self.cast([[1, 1], [1, -1]], dtype=state.dtype)
        gate = gate / self.np.sqrt(2)
        gate = self.cast(gate)
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, gate, nqubits, 0, qubits]

    def apply_x_args(self, state, nqubits, controls=[]):
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, nqubits, 0, qubits]

    def apply_y_args(self, state, nqubits, controls=[]):
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, nqubits, 0, qubits]

    def apply_z_args(self, state, nqubits, controls=[]):
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, nqubits, 0, qubits]

    def apply_z_pow_args(self, state, nqubits, controls=[]):
        gate = self.np.exp(1j * 0.1234)
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, gate, nqubits, 0, qubits]

    def apply_two_qubit_gate_args(self, state, nqubits, controls=[]):
        gate = self.np.random.random((4, 4)) + 1j * self.np.random.random((4, 4))
        gate = self.cast(gate)
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, gate, nqubits, 0, 1, qubits]

    def apply_swap_args(self, state, nqubits, controls=[]):
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, nqubits, 0, 1, qubits]

    def apply_fsim_args(self, state, nqubits, controls=[]):
        gate = list(self.np.random.random(4) + 1j * self.np.random.random(4))
        gate.append(self.np.exp(-1j * 0.1234))
        gate = self.np.array(gate, dtype="complex128")
        gate = self.cast(gate)
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, gate, nqubits, 0, 1, qubits]

    def initial_state_args(self, state, nqubits, controls=[]):
        if controls:
            raise NotImplementedError
        return [nqubits, state.dtype]

    def collapse_state_args(self, state, nqubits, controls=[]):
        if controls:
            raise NotImplementedError
        qubits = self.qubits_tensor(nqubits, [0], controls)
        return [state, qubits, 0, nqubits]

    def measure_frequencies_args(self, state, nqubits, controls=[]):
        if controls:
            raise NotImplementedError
        frequencies = self.np.zeros(state.shape, dtype="int64")
        frequencies = self.cast(frequencies, dtype="int64")
        probs = self.cast(self.np.abs(state) ** 2, dtype="float64")
        return [frequencies, probs]

    def qft_args(self, state, nqubits, controls=[]):
        return [state, nqubits]

    def __getattr__(self, x):
        return getattr(self.op, x)

    def qft(self, state, nqubits):
        import numpy as np
        matrix = self.np.array([[1, 1], [1, -1]])
        matrix = self.cast(matrix / self.np.sqrt(2), dtype=state.dtype)
        for i1 in range(nqubits):
            qubits = self.qubits_tensor(nqubits, [i1])
            state = self.apply_gate(state, matrix, nqubits, i1, qubits)
            for i2 in range(i1 + 1, nqubits):
                theta = self.cast(self.np.pi / 2 ** (i2 - i1))
                phase = np.exp(1j * theta).astype(state.dtype)
                qubits = self.qubits_tensor(nqubits, [i1], [i2])
                state = self.apply_z_pow(state, phase, nqubits, i1, qubits)

        for i in range(nqubits // 2):
            qubits = self.qubits_tensor(nqubits, [i, nqubits - i - 1])
            state = self.apply_swap(state, nqubits, i, nqubits - i - 1, qubits)

        return state
