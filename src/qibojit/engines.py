import numpy as np
from qibo.config import raise_error
from qibo.engines.abstract import Simulator
from qibojit.matrices import CustomMatrices


GATE_OPS = {
    "X": "apply_x",
    "CNOT": "apply_x",
    "TOFFOLI": "apply_x",
    "Y": "apply_y",
    "Z": "apply_z",
    "CZ": "apply_z",
    "U1": "apply_z_pow",
    "CU1": "apply_z_pow",
    "SWAP": "apply_swap",
    "fSim": "apply_fsim",
    "GeneralizedfSim": "apply_fsim"
}

def get_op(gate):
    op = GATE_OPS.get(gate.__class__.__name__)
    if op is None:
        ntargets = len(gate.target_qubits)
        if ntargets == 1:
            return "apply_gate"
        elif ntargets == 2:
            return "apply_two_qubit_gate_kernel"
        else:
            return "apply_multi_qubit_gate"
    else:
        return op


class NumbaEngine(Simulator):

    def __init__(self):
        super().__init__()
        import psutil
        from qibojit.custom_operators import gates, ops
        self.name = "qibojit"
        self.platform = "numba"
        self.device = "/CPU:0"
        self.matrices = CustomMatrices(self.dtype)
        self.gates = gates
        self.ops = ops
        self.multi_qubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel
        }
        self.set_threads(psutil.cpu_count(logical=False))

    def set_threads(self, nthreads):
        import numba
        numba.set_num_threads(nthreads)
        self.nthreads = nthreads

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        nstates = 1 << (nqubits - ncontrols - 1)
        if ncontrols:
            kernel = getattr(self.gates, "multicontrol_{}_kernel".format(kernel))
            return kernel(state, gate, qubits, nstates, m)
        kernel = getattr(self.gates, "{}_kernel".format(kernel))
        return kernel(state, gate, nstates, m)

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits):
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
            kernel = getattr(self.gates, "multicontrol_{}_kernel".format(kernel))
            return kernel(state, gate, qubits, nstates, m1, m2, swap_targets)
        kernel = getattr(self.gates, "{}_kernel".format(kernel))
        return kernel(state, gate, nstates, m1, m2, swap_targets)

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits):
        if qubits is None:
            qubits = np.array(sorted(nqubits - q - 1 for q in targets), dtype="int32")
        nstates = 1 << (nqubits - len(qubits))
        targets = np.array([1 << (nqubits - t - 1) for t in targets[::-1]], dtype="int64")
        if len(targets) > 5:
            kernel = self.gates.apply_multi_qubit_gate_kernel
        else:
            kernel = self.multi_qubit_kernels.get(len(targets))
        return kernel(state, gate, qubits, nstates, targets)

    def _create_qubits_tensor(self, gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return np.array(sorted(qubits), dtype="int32")

    def apply_gate(self, gate, state, nqubits):
        # TODO: Implement density matrices (most likely in another method)
        op = get_op(gate)
        matrix = self.asmatrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        targets = gate.target_qubits
        if len(targets) == 1:
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        elif len(targets) == 2:
            return self.two_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        else:
            return self.multi_qubit_base(state, nqubits, targets, op, matrix, qubits)

    def zero_state(self, nqubits, is_matrix=False):
        """Generate |000...0> state as an array."""
        size = 2 ** nqubits
        if is_matrix:
            state = np.empty((size, size), dtype=self.dtype)
            return self.ops.initial_density_matrix(state)
        state = np.empty((size,), dtype=self.dtype)
        return self.ops.initial_state_vector(state)


class CupyEngine(Simulator):

    DEFAULT_BLOCK_SIZE = 1024
    MAX_NUM_TARGETS = 7

    def __init__(self):
        super().__init__()
        import os
        import cupy as cp  # pylint: disable=import-error
        import cupy_backends  # pylint: disable=import-error
        self.name = "qibojit"
        self.platform = "cupy"
        self.device = "/GPU:0"
        self.kernel_type = "double"
        self.matrices = CustomMatrices(self.dtype)
        try:
            if not cp.cuda.runtime.getDeviceCount(): # pragma: no cover
                raise RuntimeError("Cannot use cupy backend if GPU is not available.")
        except cp.cuda.runtime.CUDARuntimeError:
            raise ImportError("Could not detect cupy compatible devices.")

        self.cp = cp
        self.is_hip = cupy_backends.cuda.api.runtime.is_hip
        self.KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
                        "apply_two_qubit_gate", "apply_fsim", "apply_swap")

        # load core kernels
        self.gates = {}
        from qibojit.custom_operators import raw_kernels
        def kernel_loader(name, ktype):
            code = getattr(raw_kernels, name)
            code = code.replace("T", f"thrust::complex<{ktype}>")
            gate = cp.RawKernel(code, name, ("--std=c++11",))
            self.gates[f"{name}_{ktype}"] = gate

        for ktype in ("float", "double"):
            for name in self.KERNELS:
                kernel_loader(f"{name}_kernel", ktype)
                kernel_loader(f"multicontrol_{name}_kernel", ktype)
            kernel_loader("collapse_state_kernel", ktype)
            kernel_loader("initial_state_kernel", ktype)

        # load multiqubit kernels
        name = "apply_multi_qubit_gate_kernel"
        for ntargets in range(3, self.MAX_NUM_TARGETS + 1):
            for ktype in ("float", "double"):
                code = getattr(raw_kernels, name)
                code = code.replace("T", f"thrust::complex<{ktype}>")
                code = code.replace("nsubstates", str(2 ** ntargets))
                code = code.replace("MAX_BLOCK_SIZE", str(self.DEFAULT_BLOCK_SIZE))
                gate = cp.RawKernel(code, name, ("--std=c++11",))
                self.gates[f"{name}_{ktype}_{ntargets}"] = gate

    def set_precision(self, precision):
        super().set_precision(precision)
        if self.dtype == "complex128":
            self.kernel_type = "double"
        elif self.dtype == "complex64":
            self.kernel_type = "float"

    def set_device(self, device):
        if "GPU" not in device:
            raise_error(ValueError, f"Device {device} is not available for {self} backend.")
        # TODO: Raise error if GPU is not available
        self.device = device

    def set_threads(self, nthreads):
        raise_error(RuntimeError, f"{self} does not thread setting.")

    def asmatrix(self, gate):
        matrix = super().asmatrix(gate)
        return self.cp.asarray(matrix)

    def calculate_blocks(self, nstates, block_size=DEFAULT_BLOCK_SIZE):
        """Compute the number of blocks and of threads per block.

        The total number of threads is always equal to ``nstates``, give that
        the kernels are designed to execute only one out of ``nstates`` updates.
        Therefore, the number of threads per block (``block_size``) changes also
        the total number of blocks. By default, it is set to ``self.DEFAULT_BLOCK_SIZE``.
        """
        # Compute the number of blocks so that at least ``nstates`` threads are launched
        nblocks = (nstates + block_size - 1) // block_size
        if nstates < block_size:
            nblocks = 1
            block_size = nstates
        return nblocks, block_size

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)
        if kernel in ("apply_x", "apply_y", "apply_z"):
            args = (state, tk, m)
        else:
            args = (state, tk, m, gate)

        if ncontrols:
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{self.kernel_type}")
            args += (qubits, ncontrols + 1)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{self.kernel_type}")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits):
        ncontrols = len(qubits) - 2 if qubits is not None else 0
        if target1 > target2:
            m1 = nqubits - target1 - 1
            m2 = nqubits - target2 - 1
            tk1, tk2 = 1 << m1, 1 << m2
            uk1, uk2 = tk2, tk1
        else:
            m1 = nqubits - target2 - 1
            m2 = nqubits - target1 - 1
            tk1, tk2 = 1 << m1, 1 << m2
            uk1, uk2 = tk1, tk2
        nstates = 1 << (nqubits - 2 - ncontrols)

        if kernel == "apply_swap":
            args = (state, tk1, tk2, m1, m2, uk1, uk2)
        else:
            args = (state, tk1, tk2, m1, m2, uk1, uk2, gate)
            assert state.dtype == args[-1].dtype

        if ncontrols:
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{self.kernel_type}")
            args += (qubits, ncontrols + 2)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{self.kernel_type}")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits):
        assert gate is not None
        ntargets = len(targets)
        if ntargets > self.MAX_NUM_TARGETS:
            raise ValueError(f"Number of target qubits must be <= {self.MAX_NUM_TARGETS}"
                             f" but is {ntargets}.")
        nactive = len(qubits)
        targets = self.cp.asarray(tuple(1 << (nqubits - t - 1) for t in targets[::-1]),
                                  dtype=self.cp.int64)
        nstates = 1 << (nqubits - nactive)
        nsubstates = 1 << ntargets
        nblocks, block_size = self.calculate_blocks(nstates)
        kernel = self.gates.get(f"apply_multi_qubit_gate_kernel_{self.kernel_type}_{ntargets}")
        args = (state, gate, qubits, targets, ntargets, nactive)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def _create_qubits_tensor(self, gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return self.cp.asarray(sorted(qubits), dtype=self.cp.int32)

    def apply_gate(self, gate, state, nqubits):
        # TODO: Implement density matrices (most likely in another method)
        op = get_op(gate)
        matrix = self.asmatrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        targets = gate.target_qubits
        if len(targets) == 1:
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        elif len(targets) == 2:
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        else:
            return self.multi_qubit_base(state, nqubits, targets, op, matrix, qubits)

    def zero_state(self, nqubits, is_matrix=False):
        n = 1 << nqubits
        kernel = self.gates.get(f"initial_state_kernel_{self.kernel_type}")
        if is_matrix:
            state = self.cp.zeros(n * n, dtype=self.dtype)
            kernel((1,), (1,), [state])
            self.cp.cuda.stream.get_current_stream().synchronize()
            state = state.reshape((n, n))
        else:
            state = self.cp.zeros(n, dtype=self.dtype)
            kernel((1,), (1,), [state])
            self.cp.cuda.stream.get_current_stream().synchronize()
        return state