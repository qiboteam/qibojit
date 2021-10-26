from abc import ABC, abstractmethod


class AbstractBackend(ABC):

    def __init__(self): # pragma: no cover
        self.name = "abstract"
        self.gates = None
        self.ops = None

    @abstractmethod
    def cast(self, x, dtype=None): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self, x): # pragma: no cover
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
    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def swap_pieces(self, piece0, piece1, new_global, nlocal): # pragma: no cover
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
        self.multiqubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel
            }

    def cast(self, x, dtype=None):
        if not isinstance(x, self.np.ndarray):
            x = self.np.array(x)
        if dtype and x.dtype != dtype:
            return x.astype(dtype)
        return x

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        return self.np.array(x)

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

    def multi_qubit_base(self, state, nqubits, targets, qubits=None, gate=None):
        if qubits is None:
            qubits = tuple(sorted(nqubits - q - 1 for q in targets))
        nstates = 1 << (nqubits - len(qubits))
        targets = tuple(1 << (nqubits - t - 1) for t in targets[::-1])
        if len(targets) > 5:
            kernel = self.gates.apply_multi_qubit_gate_kernel
        else:
            kernel = self.multiqubit_kernels.get(len(targets))
        return kernel(state, gate, qubits, nstates, targets)

    def initial_state(self, nqubits, dtype, is_matrix=False):
        if isinstance(dtype, str):
            dtype = getattr(self.np, dtype)
        size = 2 ** nqubits
        if is_matrix:
            state = self.np.empty((size, size), dtype=dtype)
            return self.ops.initial_density_matrix(state)
        state = self.np.empty((size,), dtype=dtype)
        return self.ops.initial_state_vector(state)

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        if normalize:
            return self.ops.collapse_state_normalized(state, qubits, result, nqubits)
        return self.ops.collapse_state(state, qubits, result, nqubits)

    def transpose_state(self, pieces, state, nqubits, order):
        return self.ops.transpose_state(tuple(pieces), state, nqubits, tuple(order))

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        return self.ops.swap_pieces(piece0, piece1, new_global, nlocal)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234, nthreads=None):
        if nthreads is None:
            import psutil
            nthreads = psutil.cpu_count(logical=False)
        return self.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed, nthreads)


class CupyBackend(AbstractBackend): # pragma: no cover

    DEFAULT_BLOCK_SIZE = 1024
    KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
               "apply_two_qubit_gate", "apply_fsim", "apply_swap")
    MULTIQUBIT_KERNELS = {
        3: "apply_three_qubit_gate_kernel",
        4: "apply_four_qubit_gate_kernel",
        5: "apply_five_qubit_gate_kernel"
    }

    def __init__(self):
        import os
        import numpy as np
        import cupy as cp  # pylint: disable=import-error
        try:
            if not cp.cuda.runtime.getDeviceCount(): # pragma: no cover
                raise RuntimeError("Cannot use cupy backend if GPU is not available.")
        except cp.cuda.runtime.CUDARuntimeError:
            raise ImportError("Could not detect cupy compatible devices.")

        self.name = "cupy"
        self.np = np
        self.cp = cp
        base_dir = os.path.dirname(os.path.realpath(__file__))

        self.kernel_double_suffix = "<thrust::complex<double> >"
        self.kernel_float_suffix = "<thrust::complex<float> >"

        # load gate kernels
        kernels = []
        for kernel in self.KERNELS:
            kernels.append(f"{kernel}_kernel{self.kernel_double_suffix}")
            kernels.append(f"{kernel}_kernel{self.kernel_float_suffix}")
            kernels.append(f"multicontrol_{kernel}_kernel{self.kernel_double_suffix}")
            kernels.append(f"multicontrol_{kernel}_kernel{self.kernel_float_suffix}")
        for ntargets in self.MULTIQUBIT_KERNELS:
            kernels.append(self.MULTIQUBIT_KERNELS.get(ntargets)+self.kernel_double_suffix)
            kernels.append(self.MULTIQUBIT_KERNELS.get(ntargets)+self.kernel_float_suffix)
        kernels.append(f"apply_multi_qubit_gate_kernel{self.kernel_double_suffix}")
        kernels.append(f"apply_multi_qubit_gate_kernel{self.kernel_float_suffix}")
        kernels.append(f"collapse_state_kernel{self.kernel_double_suffix}")
        kernels.append(f"collapse_state_kernel{self.kernel_float_suffix}")
        kernels.append(f"initial_state_kernel{self.kernel_double_suffix}")
        kernels.append(f"initial_state_kernel{self.kernel_float_suffix}")
        kernels = tuple(kernels)
        gates_dir = os.path.join(base_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            code = r"{}".format(file.read())
            self.gates = cp.RawModule(code=code, options=("--std=c++11",),
                                      name_expressions=kernels)

    def calculate_blocks(self, nstates, block_size=None):
        """Compute the number of blocks and of threads per block.

        The total number of threads is always equal to ``nstates``, give that
        the kernels are designed to execute only one out of ``nstates`` updates.
        Therefore, the number of threads per block (``block_size``) changes also
        the total number of blocks. By default, it is set to ``self.DEFAULT_BLOCK_SIZE``.
        """
        # Set default value for block_size
        if block_size is None:
            block_size = self.DEFAULT_BLOCK_SIZE

        # Compute the number of blocks so that at least ``nstates`` threads are launched
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
        if isinstance(x, self.np.ndarray):
            return x
        return x.get()

    def get_kernel_type(self, state):
        if state.dtype == self.cp.complex128:
            return self.kernel_double_suffix
        elif state.dtype == self.cp.complex64:
            return self.kernel_float_suffix
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
            args = (state, tk, m, self.cast(gate, dtype=state.dtype).flatten())

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel{ktype}")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 1)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel{ktype}")

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
            args = (state, tk1, tk2, m1, m2, uk1, uk2, self.cast(gate).flatten())
            assert state.dtype == args[-1].dtype

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel{ktype}")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 2)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel{ktype}")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def multi_qubit_base(self, state, nqubits, targets, qubits=None, gate=None):
        assert gate is not None
        state = self.cast(state)
        gate = self.cast(gate.flatten())
        assert state.dtype == gate.dtype

        ntargets = len(targets)
        if qubits is None:
            nactive = ntargets
            qubits = self.cast(sorted(nqubits - q - 1 for q in targets), dtype=self.cp.int32)
        else:
            nactive = len(qubits)
            qubits = self.cast(qubits, dtype=self.cp.int32)
        targets = self.cast(tuple(1 << (nqubits - t - 1) for t in targets[::-1]),
                            dtype=self.cp.int64)
        nstates = 1 << (nqubits - nactive)
        nsubstates = 1 << ntargets

        # If len(targets) is 3,4 or 5 we can use hard-coded kernels,
        # otherwise we need to call general multi-qubit gate kernels.
        # The latter require a full copy ``buffer``` of the state vector
        ktype = self.get_kernel_type(state)
        if len(targets) < 6:
            # Compute the number of blocks and threads
            # To avoid memory issues, reduce the block size with len(targets)
            # len(targets): 3 -> 1024 threads per block
            # len(targets): 4 -> 512  threads per block
            # len(targets): 5 -> 256  threads per block
            nblocks, block_size = self.calculate_blocks(nstates,
                                                        block_size=2**(13-len(targets)))
            kernel = self.gates.get_function(self.MULTIQUBIT_KERNELS.get(len(targets))+ktype)
            args = (state, gate, qubits, targets, ntargets, nactive)
        else:
            nblocks, block_size = self.calculate_blocks(nstates)
            kernel = self.gates.get_function(f"apply_multi_qubit_gate_kernel{ktype}")
            buffer = self.cp.copy(state) # full copy of the state vector, to be used as buffer
            args = (state, buffer, gate, qubits, targets, nsubstates, ntargets, nactive)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def initial_state(self, nqubits, dtype, is_matrix=False):
        n = 1 << nqubits
        if dtype in {"complex128", self.np.complex128, self.cp.complex128}:
            ktype = self.kernel_double_suffix
        elif dtype in {"complex64", self.np.complex64, self.cp.complex64}:
            ktype = self.kernel_float_suffix
        else: # pragma: no cover
            raise TypeError("Unknown dtype {} passed in initial state operator."
                            "".format(dtype))
        kernel = self.gates.get_function(f"initial_state_kernel{ktype}")

        if is_matrix:
            state = self.cp.zeros(n * n, dtype=dtype)
            kernel((1,), (1,), [state])
            state = state.reshape((n, n))
        else:
            state = self.cp.zeros(n, dtype=dtype)
            kernel((1,), (1,), [state])
        return state

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        nblocks, block_size = self.calculate_blocks(nstates)

        state = self.cast(state)
        ktype = self.get_kernel_type(state)
        args = [state, self.cast(qubits, dtype=self.cp.int32), result, ntargets]
        kernel = self.gates.get_function(f"collapse_state_kernel{ktype}")
        kernel((nblocks,), (block_size,), args)

        if normalize:
            norm = self.cp.sqrt(self.cp.sum(self.cp.square(self.cp.abs(state))))
            state = state / norm
        return state

    def transpose_state(self, pieces, state, nqubits, order):
        raise NotImplementedError("`transpose_state` method is not "
                                  "implemented for GPU.")

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        raise NotImplementedError("`swap_pieces` method is not "
                                  "implemented for GPU.")

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        raise NotImplementedError("`measure_frequencies` method is not "
                                  "implemented for GPU.")
