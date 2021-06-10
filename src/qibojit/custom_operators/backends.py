from abc import ABC, abstractmethod


class AbstractBackend(ABC):

    def __init__(self): # pragma: no cover
        self.name = "abstract"
        self.gates = None
        self.ops = None

    def cast(self, x, dtype=None):
        return x

    def to_numpy(self, x):
        return x

    @abstractmethod
    def free_all_blocks(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, nqubits, dtype, is_matrix=False): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def collapse_state(self, state, qubits, result, nqubits, normalize): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234): # pragma: no cover
        raise NotImplementedError


class NumbaBackend(AbstractBackend):

    def __init__(self):
        import numpy as np
        from qibojit.custom_operators import gates, ops
        self.name = "numba"
        self.gates = gates
        self.ops = ops
        self.np = np

    def cast(self, x, dtype=None):
        if not isinstance(x, self.np.ndarray):
            x = self.np.array(x)
        if dtype:
            return x.astype(dtype)
        return x

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        return self.np.array(x)

    def free_all_blocks(self):
        pass

    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        nstates = 1 << (nqubits - ncontrols - 1)
        if ncontrols:
            kernel = getattr(self.gates, "multicontrol_{}_kernel".format(kernel))
            return kernel(state, gate, qubits, nstates, m)
        kernel = getattr(self.gates, "{}_kernel".format(kernel))
        return kernel(state, gate, nstates, m)

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
        if ncontrols:
            kernel = getattr(self.gates, "multicontrol_{}_kernel".format(kernel))
            return kernel(state, gate, qubits, nstates, m1, m2, swap_targets)
        kernel = getattr(self.gates, "{}_kernel".format(kernel))
        return kernel(state, gate, nstates, m1, m2, swap_targets)

    def initial_state(self, nqubits, dtype, is_matrix=False):
        if isinstance(dtype, str):
            dtype = getattr(self.np, dtype)
        if is_matrix:
            return self.ops.initial_density_matrix(nqubits, dtype)
        return self.ops.initial_state_vector(nqubits, dtype)

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        if normalize:
            return self.ops.collapse_state_normalized(state, qubits, result, nqubits)
        return self.ops.collapse_state(state, qubits, result, nqubits)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234, nthreads=None):
        if nthreads is None:
            import psutil
            nthreads = psutil.cpu_count(logical=False)
        return self.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed, nthreads)


class CupyBackend(AbstractBackend): # pragma: no cover

    DEFAULT_BLOCK_SIZE = 1024
    KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
               "apply_two_qubit_gate", "apply_fsim", "apply_swap")

    def __init__(self):
        import os
        import cupy as cp
        try:
            if not cp.cuda.runtime.getDeviceCount(): # pragma: no cover
                raise RuntimeError("Cannot use cupy backend if GPU is not available.")
        except cp.cuda.runtime.CUDARuntimeError:
            raise ImportError("Could not detect cupy compatible devices.")

        self.name = "cupy"
        self.cp = cp
        base_dir = os.path.dirname(os.path.realpath(__file__))

        # load gate kernels
        kernels = []
        for kernel in self.KERNELS:
            kernels.append(f"{kernel}_kernel<complex<double>>")
            kernels.append(f"{kernel}_kernel<complex<float>>")
            kernels.append(f"multicontrol_{kernel}_kernel<complex<double>>")
            kernels.append(f"multicontrol_{kernel}_kernel<complex<float>>")
        kernels.append("collapse_state_kernel<complex<double>>")
        kernels.append("collapse_state_kernel<complex<float>>")
        kernels = tuple(kernels)
        gates_dir = os.path.join(base_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            code = r"{}".format(file.read())
            self.gates = cp.RawModule(code=code, options=("--std=c++11",),
                                      name_expressions=kernels)

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

    def free_all_blocks(self):
        self.cp._default_memory_pool.free_all_blocks()

    def get_kernel_type(self, state):
        if state.dtype == self.cp.complex128:
            return "complex<double>"
        elif state.dtype == self.cp.complex64:
            return "complex<float>"
        raise TypeError("State of invalid type {}.".format(state.dtype))

    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)

        state = self.cast(state)
        if gate is None:
            args = (state, tk, m)
        else:
            args = (state, tk, m, self.cast(gate, dtype=state.dtype))

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel<{ktype}>")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 1)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel<{ktype}>")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None):
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

        state = self.cast(state)
        if gate is None:
            args = (state, tk1, tk2, m1, m2, uk1, uk2)
        else:
            args = (state, tk1, tk2, m1, m2, uk1, uk2, self.cast(gate))
            assert state.dtype == args[-1].dtype

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel<{ktype}>")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 2)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel<{ktype}>")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def initial_state(self, nqubits, dtype, is_matrix=False):
        n = 1 << nqubits
        if is_matrix:
            state = self.cp.zeros((n, n), dtype=dtype)
            state[0, 0] = 1
        else:
            state = self.cp.zeros(n, dtype=dtype)
            state[0] = 1
        return state

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        nblocks, block_size = self.calculate_blocks(nstates)

        state = self.cast(state)
        ktype = self.get_kernel_type(state)
        args = [state, self.cast(qubits, dtype=self.cp.int32),
                self.cast(result, dtype=self.cp.int32), ntargets]
        kernel = self.gates.get_function(f"collapse_state_kernel<{ktype}>")
        kernel((nblocks,), (block_size,), args)

        if normalize:
            norm = self.cp.sqrt(self.cp.sum(self.cp.square(self.cp.abs(state))))
            state = state / norm

        return state

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        raise NotImplementedError("`measure_frequencies` method is not "
                                  "implemented for GPU.")