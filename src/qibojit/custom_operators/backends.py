from abc import ABC, abstractmethod


class AbstractBackend(ABC):

    def __init__(self): # pragma: no cover
        self.name = "abstract"
        self.gates = None
        self.ops = None
        self.test_regressions = {}

    @abstractmethod
    def cast(self, x, dtype=None): # pragma: no cover
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
        self.multi_qubit_kernels = {
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
            qubits = self.np.array(sorted(nqubits - q - 1 for q in targets), dtype="int32")
        nstates = 1 << (nqubits - len(qubits))
        targets = self.np.array([1 << (nqubits - t - 1) for t in targets[::-1]], dtype="int64")
        if len(targets) > 5:
            kernel = self.gates.apply_multi_qubit_gate_kernel
        else:
            kernel = self.multi_qubit_kernels.get(len(targets))
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
        return self.ops.transpose_state(tuple(pieces), state, nqubits, order)

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        return self.ops.swap_pieces(piece0, piece1, new_global, nlocal)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234, nthreads=None):
        if nthreads is None:
            import psutil
            nthreads = psutil.cpu_count(logical=False)
        return self.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed, nthreads)


class CupyBackend(AbstractBackend): # pragma: no cover
    # CI does not test for GPU

    DEFAULT_BLOCK_SIZE = 1024
    MAX_NUM_TARGETS = 7

    def __init__(self):
        import os
        import numpy as np
        import cupy as cp  # pylint: disable=import-error
        import cupy_backends  # pylint: disable=import-error
        try:
            if not cp.cuda.runtime.getDeviceCount(): # pragma: no cover
                raise RuntimeError("Cannot use cupy backend if GPU is not available.")
        except cp.cuda.runtime.CUDARuntimeError:
            raise ImportError("Could not detect cupy compatible devices.")

        self.name = "cupy"
        self.np = np
        self.cp = cp
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.is_hip = cupy_backends.cuda.api.runtime.is_hip
        self.KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
                        "apply_two_qubit_gate", "apply_fsim", "apply_swap")
        self.kernel_double_suffix = "<thrust::complex<double> >"
        self.kernel_float_suffix = "<thrust::complex<float> >"
        if self.is_hip:  # pragma: no cover
            self.test_regressions = {
                "test_measurementresult_apply_bitflips": [
                    [2, 2, 6, 1, 0, 0, 0, 0, 1, 0],
                    [2, 2, 6, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 4, 1, 0, 0, 0, 0, 1, 0],
                    [0, 2, 4, 0, 0, 0, 0, 0, 0, 0]
                ],
                "test_probabilistic_measurement": {2: 267, 3: 247, 0: 243, 1: 243},
                "test_unbalanced_probabilistic_measurement": {3: 500, 2: 174, 0: 163, 1: 163},
                "test_post_measurement_bitflips_on_circuit": [
                    {5: 30}, {5: 17, 7: 7, 1: 2, 4: 2, 2: 1, 3: 1},
                    {7: 7, 1: 5, 3: 4, 6: 4, 2: 3, 5: 3, 0: 2, 4: 2}
                ]
            }
        else:  # pragma: no cover
            self.test_regressions = {
                "test_measurementresult_apply_bitflips": [
                    [0, 0, 0, 6, 4, 1, 1, 4, 0, 2],
                    [0, 0, 0, 6, 4, 1, 1, 4, 0, 2],
                    [0, 0, 0, 0, 4, 1, 1, 4, 0, 0],
                    [0, 0, 0, 6, 4, 0, 0, 4, 0, 2]
                ],
                "test_probabilistic_measurement": {0: 264, 1: 235, 2: 269, 3: 232},
                "test_unbalanced_probabilistic_measurement": {0: 170, 1: 154, 2: 167, 3: 509},
                "test_post_measurement_bitflips_on_circuit": [
                    {5: 30}, {5: 12, 7: 7, 6: 5, 4: 3, 1: 2, 2: 1},
                    {2: 10, 6: 5, 5: 4, 0: 3, 7: 3, 1: 2, 3: 2, 4: 1}
                ]
            }

        # load gate kernels
        kernels = []
        for kernel in self.KERNELS:
            kernels.append(f"{kernel}_kernel{self.kernel_double_suffix}")
            kernels.append(f"{kernel}_kernel{self.kernel_float_suffix}")
            kernels.append(f"multicontrol_{kernel}_kernel{self.kernel_double_suffix}")
            kernels.append(f"multicontrol_{kernel}_kernel{self.kernel_float_suffix}")
        for ntargets in range(3, self.MAX_NUM_TARGETS+1):
            kernels.append(f"apply_multi_qubit_gate_kernel{self.kernel_double_suffix[0:-2]}, {2**ntargets}>")
            kernels.append(f"apply_multi_qubit_gate_kernel{self.kernel_float_suffix[0:-2]}, {2**ntargets}>")
        kernels.append(f"collapse_state_kernel{self.kernel_double_suffix}")
        kernels.append(f"collapse_state_kernel{self.kernel_float_suffix}")
        kernels.append(f"initial_state_kernel{self.kernel_double_suffix}")
        kernels.append(f"initial_state_kernel{self.kernel_float_suffix}")
        kernels = tuple(kernels)
        gates_dir = os.path.join(base_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            code = r"{}".format(file.read())
            code = code.replace("QIBO_MAX_BLOCK_SIZE", str(self.DEFAULT_BLOCK_SIZE))
            self.gates = cp.RawModule(code=code, options=("--std=c++11",),
                                      name_expressions=kernels)
        self.gates.compile()

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

    def cast(self, x, dtype=None):
        if isinstance(x, self.cp.ndarray):
            return x
        return self.cp.asarray(x, dtype=dtype)

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
        if ntargets > self.MAX_NUM_TARGETS:
            raise ValueError(f"Number of target qubits must be <= {self.MAX_NUM_TARGETS}"
                             f" but is {ntargets}.")
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

        ktype = self.get_kernel_type(state)
        nblocks, block_size = self.calculate_blocks(nstates)
        kernel = self.gates.get_function(f"apply_multi_qubit_gate_kernel{ktype[0:-2]}, {nsubstates}>")
        args = (state, gate, qubits, targets, ntargets, nactive)
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

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234, nthreads=None):
        raise NotImplementedError("`measure_frequencies` method is not "
                                  "implemented for GPU.")
