class AbstractBackend:

    def __init__(self):
        import numpy as np
        self.np = np
        self.op = None

    def cast(self, x, dtype=None):
        if dtype is not None:
            return self.np.array(x, dtype=dtype)
        return x

    def to_numpy(self, x, dtype="complex128"):
        if dtype is not None:
            return x.astype(dtype)
        return x

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = [nqubits - q - 1 for q in targets]
        qubits.extend(nqubits - q - 1 for q in controls)
        return sorted(qubits)

    def apply_gate_args(self, state, nqubits, controls=[]):
        gate = self.cast([[1, 1], [1, -1]], dtype=state.dtype)
        gate = gate / self.np.sqrt(2)
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
        qubits = self.np.self.qubits_tensor(nqubits, [0], controls)
        return [state, gate, nqubits, 0, qubits]

    def apply_two_qubit_gate_args(self, state, nqubits, controls=[]):
        gate = self.np.random.random((4, 4)) + 1j * self.np.random.random((4, 4))
        gate = gate.astype(state.dtype)
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, gate, nqubits, 0, 1, qubits]

    def apply_swap_args(self, state, nqubits, controls=[]):
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, nqubits, 0, 1, qubits]

    def apply_fsim_args(self, state, nqubits, controls=[]):
        gate = self.np.random.random(4) + 1j * self.np.random.random(4)
        gate = gate.astype(state.dtype)
        phase = self.np.array([np.exp(-1j * 0.1234)], dtype=state.dtype)
        gate = self.np.concatenate([gate, phase])
        qubits = self.qubits_tensor(nqubits, [0, 1], controls)
        return [state, gate, nqubits, 0, 1, qubits]

    def initial_state_args(self, state, nqubits, controls=[]):
        return [nqubits, state.dtype]

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
                qubits = self.qubits_tensor(nqubits, [i1], [i2])
                state = self.apply_z_pow(state, theta, nqubits, i1, qubits)

        for i in range(nqubits // 2):
            qubits = self.qubits_tensor(nqubits, [i, nqubits - i - 1])
            state = self.apply_swap(state, nqubits, i, nqubits - i - 1, qubits)

        return state
