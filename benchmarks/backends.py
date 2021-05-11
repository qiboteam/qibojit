from abstract import AbstractBackend


class NumbaBackend(AbstractBackend):

    def __init__(self):
        from qibojit import custom_operators as op
        self.op = op

    def qubits_tensor(self, nqubits, targets, controls=[]):
        return tuple(super().qubits_tensor(nqubits, targets, controls))


class TensorflowBackend(AbstractBackend):

    def __init__(self):
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import numpy as np
        import tensorflow as tf
        from qibotf import custom_operators as op
        self.np = np
        self.backend = tf
        self.op = op

    def cast(self, x, dtype="complex128"):
        return self.backend.cast(x, dtype=dtype)

    def to_numpy(self, x):
        return self.np.array(x)

    def qubits_tensor(self, nqubits, targets, controls=[]):
        qubits = super().qubits_tensor(nqubits, targets, controls)
        return self.cast(qubits, dtype=self.backend.int32)


class Backends(dict):

    _implemented = {"numba": NumbaBackend, "qibotf": TensorflowBackend}

    def get(self, name):
        if name not in self:
            if name not in self._implemented:
                raise KeyError("Unknown backend {}.".format(name))
            self[name] = self._implemented.get(name)()
        return super().get(name)


backends = Backends()
