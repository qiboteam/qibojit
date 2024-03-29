from abstract import AbstractBackend


class NumbaBackend(AbstractBackend):
    def __init__(self):
        super().__init__()
        from qibojit import custom_operators as op

        self.op = op

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = super().qubits_tensor(nqubits, targets, controls)
        return qubits


class CupyBackend(AbstractBackend):
    def __init__(self):
        super().__init__()
        from qibojit import custom_operators as op

        op.set_backend("cupy")
        self.op = op

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = super().qubits_tensor(nqubits, targets, controls)
        return self.cast(qubits, dtype="int32")


class TensorflowBackend(AbstractBackend):
    def __init__(self):
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        super().__init__()
        import tensorflow as tf
        from qibotf import custom_operators as op

        self.backend = tf
        self.op = op
        import psutil

        self.nthreads = psutil.cpu_count(logical=False)

    def cast(self, x, dtype="complex128"):
        return self.backend.cast(x, dtype=dtype)

    def to_numpy(self, x):
        return self.np.array(x)

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = super().qubits_tensor(nqubits, targets, controls)
        return self.cast(qubits, dtype="int32")

    def apply_gate(self, state, gate, nqubits, target, qubits):
        return self.op.apply_gate(state, gate, qubits, nqubits, target, self.nthreads)

    def apply_x(self, state, nqubits, target, qubits):
        return self.op.apply_x(state, qubits, nqubits, target, self.nthreads)

    def apply_y(self, state, nqubits, target, qubits):
        return self.op.apply_y(state, qubits, nqubits, target, self.nthreads)

    def apply_z(self, state, nqubits, target, qubits):
        return self.op.apply_z(state, qubits, nqubits, target, self.nthreads)

    def apply_z_pow(self, state, gate, nqubits, target, qubits):
        return self.op.apply_z_pow(state, gate, qubits, nqubits, target, self.nthreads)

    def apply_two_qubit_gate(self, state, gate, nqubits, target1, target2, qubits):
        return self.op.apply_two_qubit_gate(
            state, gate, qubits, nqubits, target1, target2, self.nthreads
        )

    def apply_swap(self, state, nqubits, target1, target2, qubits):
        return self.op.apply_swap(
            state, qubits, nqubits, target1, target2, self.nthreads
        )

    def apply_fsim(self, state, gate, nqubits, target1, target2, qubits):
        return self.op.apply_fsim(
            state, gate, qubits, nqubits, target1, target2, self.nthreads
        )

    def initial_state(self, nqubits, dtype, is_matrix=False):
        return self.op.initial_state(nqubits, dtype, is_matrix, self.nthreads)

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        return self.op.collapse_state(
            state, qubits, result, nqubits, normalize, self.nthreads
        )

    def collapse_state_args(self, state, nqubits, controls=[]):
        if controls:
            raise NotImplementedError
        qubits = self.qubits_tensor(nqubits, [0], controls)
        result = self.cast([0], dtype="int64")
        return [state, qubits, result, nqubits]

    def measure_frequencies(self, frequencies, probs, nshots, nqubits):
        return self.op.measure_frequencies(
            frequencies, probs, nshots, nqubits, 1234, self.nthreads
        )


class Backends(dict):
    _implemented = {
        "numba": NumbaBackend,
        "cupy": CupyBackend,
        "tensorflow": TensorflowBackend,
    }

    def get(self, name):
        if name not in self:
            if name not in self._implemented:
                raise KeyError("Unknown backend {}.".format(name))
            self[name] = self._implemented.get(name)()
        return super().get(name)


backends = Backends()
