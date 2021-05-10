class NumbaBackend:

    def __init__(self):
        from qibojit import custom_operators as op
        self.op = op

    def cast(self, x):
        return x

    def to_numpy(self, x):
        return x

    @staticmethod
    def qubits_tensor(nqubits, targets, controls=[]):
        qubits = [nqubits - q - 1 for q in targets]
        qubits.extend(nqubits - q - 1 for q in controls)
        return tuple(sorted(qubits))

    def __getattr__(self, x):
        return getattr(self.op, x)

    def qft(self, state, nqubits):
        import numpy as np
        matrix = np.array([[1, 1], [1, -1]]).astype("complex128")
        matrix = self.cast(matrix / np.sqrt(2))
        for i1 in range(nqubits):
            qubits = self.qubits_tensor(nqubits, [i1])
            state = self.apply_gate(state, matrix, nqubits, i1, qubits)
            for i2 in range(i1 + 1, nqubits):
                theta = self.cast(np.pi / 2 ** (i2 - i1))
                qubits = self.qubits_tensor(nqubits, [i1], [i2])
                state = self.apply_z_pow(state, theta, nqubits, i1, qubits)

        for i in range(nqubits // 2):
            qubits = self.qubits_tensor(nqubits, [i, nqubits - i - 1])
            state = self.apply_swap(state, nqubits, i, nqubits - i - 1, qubits)

        return state


class Backends(dict):

    _implemented = {"numba": NumbaBackend}

    def get(self, name):
        if name not in self:
            if name not in self._implemented:
                raise KeyError("Unknown backend {}.".format(name))
            self[name] = self._implemented.get(name)()
        return super().get(name)


backends = Backends()
