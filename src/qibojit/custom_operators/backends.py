from abc import ABC, abstractmethod


class AbstractBackend(ABC):

    def __init__(self):
        self.name = "abstract"
        self.gates = None
        self.ops = None

    def cast(self, x, dtype=None):
        return x

    def to_numpy(self, x):
        return x

    @abstractmethod
    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        raise NotImplementedError

    @abstractmethod
    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None):
        raise NotImplementedError


class NumbaBackend(AbstractBackend):

    def __init__(self):
        from qibojit.custom_operators import gates, ops
        self.name = "numba"
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


class CupyBackend(AbstractBackend):

    DEFAULT_BLOCK_SIZE = 1024

    def __init__(self):
        import os
        import cupy as cp
        self.name = "cupy"
        self.cp = cp
        gates_dir = os.path.dirname(os.path.realpath(__file__))
        gates_dir = os.path.join(gates_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            self.gates = cp.RawModule(code=r"{}".format(file.read()))

        self.ops = None # TODO: Implement this

    def calculate_blocks(self, nstates):
        block_size = self.DEFAULT_BLOCK_SIZE
        nblocks = (nstates + block_size - 1) // block_size
        if nstates < block_size:
            nblocks = 1
            block_size = nstates
        return nblocks, block_size

    def cast(self, x, dtype=None):
        if isinstance(x, self.cp.ndarray):
            return x
        return self.cp.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return x.get()

    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)

        state = self.cast(state)
        if gate is None:
            args = (state, tk, m)
        else:
            args = (state, tk, m, self.cast(gate))

        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 1)
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

        state = self.cast(state)
        if gate is None:
            args = (state, tk1, tk2, m1, m2)
        else:
            args = (state, tk1, tk2, m1, m2, self.cast(gate))

        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 2)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state
