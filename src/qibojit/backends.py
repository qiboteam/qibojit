import numpy as np
from qibo.config import raise_error, log
from qibo.gates.abstract import ParametrizedGate
from qibo.gates.special import FusedGate
from qibo.backends.numpy import NumpyBackend
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


class NumbaBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        import psutil
        from qibojit.custom_operators import gates, ops
        self.name = "qibojit"
        self.platform = "numba"
        self.numeric_types = (int, float, complex, np.int32,
                              np.int64, np.float32, np.float64,
                              np.complex64, np.complex128)
        self.tensor_types = (np.ndarray,)
        self.device = "/CPU:0"
        self.custom_matrices = CustomMatrices(self.dtype)
        self.gates = gates
        self.ops = ops
        self.measure_frequencies_op = ops.measure_frequencies
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

    #def cast(self, x, dtype=None, copy=False): Inherited from ``NumpyBackend``

    #def to_numpy(self, x): Inherited from ``NumpyBackend``

    def zero_state(self, nqubits):
        size = 2 ** nqubits
        state = np.empty((size,), dtype=self.dtype)
        return self.ops.initial_state_vector(state)

    def zero_density_matrix(self, nqubits):
        size = 2 ** nqubits
        state = np.empty((size, size), dtype=self.dtype)
        return self.ops.initial_density_matrix(state)

    #def asmatrix_special(self, gate): Inherited from ``NumpyBackend``

    #def control_matrix(self, gate): Inherited from ``NumpyBackend``

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

    @staticmethod
    def _create_qubits_tensor(gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return np.array(sorted(qubits), dtype="int32")

    def _as_custom_matrix(self, gate):
        name = gate.__class__.__name__
        if isinstance(gate, ParametrizedGate):
            return getattr(self.custom_matrices, name)(*gate.parameters)
        elif isinstance(gate, FusedGate):
            return self.asmatrix_fused(gate)
        else:
            return getattr(self.custom_matrices, name)

    def apply_gate(self, gate, state, nqubits):
        matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        targets = gate.target_qubits
        state = self.cast(state)
        if len(targets) == 1:
            op = GATE_OPS.get(gate.__class__.__name__, "apply_gate")
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        elif len(targets) == 2:
            op = GATE_OPS.get(gate.__class__.__name__, "apply_two_qubit_gate")
            return self.two_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        else:
            return self.multi_qubit_base(state, nqubits, targets, matrix, qubits)

    def apply_gate_density_matrix(self, gate, state, nqubits, inverse=False):
        name = gate.__class__.__name__
        if name == "Y":
            return self._apply_ygate_density_matrix(gate, state, nqubits)
        if inverse:
            # used to reset the state when applying channels
            # see :meth:`qibojit.backend.NumpyBackend.apply_channel_density_matrix` below
            matrix = np.linalg.inv(gate.asmatrix(self))
            matrix = self.cast(matrix)
        else:
            matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        qubits_dm = qubits + nqubits
        targets = gate.target_qubits
        targets_dm = tuple(q + nqubits for q in targets)

        state = self.cast(state)
        shape = state.shape
        if len(targets) == 1:
            op = GATE_OPS.get(name, "apply_gate")
            state = self.one_qubit_base(state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm)
            state = self.one_qubit_base(state, 2 * nqubits, *targets_dm, op, np.conj(matrix), qubits)
        elif len(targets) == 2:
            op = GATE_OPS.get(name, "apply_two_qubit_gate")
            state = self.two_qubit_base(state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm)
            state = self.two_qubit_base(state, 2 * nqubits, *targets_dm, op, np.conj(matrix), qubits)
        else:
            state = self.multi_qubit_base(state.ravel(), 2 * nqubits, targets, matrix, qubits_dm)
            state = self.multi_qubit_base(state, 2 * nqubits, targets_dm, np.conj(matrix), qubits)
        return np.reshape(state, shape)

    def _apply_ygate_density_matrix(self, gate, state, nqubits):
        matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        qubits_dm = qubits + nqubits
        targets = gate.target_qubits
        targets_dm = tuple(q + nqubits for q in targets)
        state = self.cast(state)
        shape = state.shape
        state = self.one_qubit_base(state.ravel(), 2 * nqubits, *targets, "apply_y", matrix, qubits_dm)
        # force using ``apply_gate`` kernel so that conjugate is properly applied
        state = self.one_qubit_base(state, 2 * nqubits, *targets_dm, "apply_gate", np.conj(matrix), qubits)
        return np.reshape(state, shape)

    #def apply_channel(self, gate): Inherited from ``NumpyBackend``

    def apply_channel_density_matrix(self, channel, state, nqubits):
        state = self.cast(state)
        new_state = (1 - channel.coefficient_sum) * state
        for coeff, gate in zip(channel.coefficients, channel.gates):
            state = self.apply_gate_density_matrix(gate, state, nqubits)
            new_state += coeff * state
            # reset the state
            state = self.apply_gate_density_matrix(gate, state, nqubits, inverse=True)
        return new_state

    #def calculate_probabilities(self, state, qubits, nqubits): Inherited from ``NumpyBackend``

    #def sample_shots(self, probabilities, nshots): Inherited from ``NumpyBackend``

    #def aggregate_shots(self, shots): Inherited from ``NumpyBackend``

    #def samples_to_binary(self, samples, nqubits): Inherited from ``NumpyBackend``

    #def samples_to_decimal(self, samples, nqubits): Inherited from ``NumpyBackend``

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probabilities, nshots)

        import collections
        seed = np.random.randint(0, int(1e8), dtype="int64")
        nqubits = int(np.log2(tuple(probabilities.shape)[0]))
        frequencies = np.zeros(2 ** nqubits, dtype="int64")
        # always fall back to numba CPU backend because for ops not implemented on GPU
        frequencies = self.measure_frequencies_op(
            frequencies, probabilities, nshots, nqubits, seed, self.nthreads)
        return collections.Counter({i: f for i, f in enumerate(frequencies) if f > 0})

    #def calculate_frequencies(self, samples): Inherited from ``NumpyBackend``

    #def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): Inherited from ``NumpyBackend``


class CupyBackend(NumpyBackend):

    DEFAULT_BLOCK_SIZE = 1024
    MAX_NUM_TARGETS = 7

    def __init__(self):
        NumpyBackend.__init__(self)
        import os
        import cupy as cp  # pylint: disable=import-error
        import cupy_backends  # pylint: disable=import-error
        self.name = "qibojit"
        self.platform = "cupy"
        self.numeric_types = (int, float, complex, cp.int32,
                              cp.int64, cp.float32, cp.float64,
                              cp.complex64, cp.complex128)
        self.tensor_types = (np.ndarray, cp.ndarray)
        from scipy import sparse
        self.npsparse = sparse
        self.sparse = cp.sparse
        self.device = "/GPU:0"
        self.kernel_type = "double"
        self.custom_matrices = CustomMatrices(self.dtype)
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

        # load numba op for measuring frequencies
        from qibojit.custom_operators.ops import measure_frequencies
        self.measure_frequencies_op = measure_frequencies

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

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if self.sparse.issparse(x):
            if dtype != x.dtype:
                return x.astype(dtype)
            else:
                return x
        elif self.npsparse.issparse(x):
            cls = getattr(self.sparse, x.__class__.__name__)
            return cls(x, dtype=dtype)
        elif isinstance(x, self.cp.ndarray):
            if copy:
                self.cp.copy(self.cp.asarray(x, dtype=dtype))
            else:
                self.cp.asarray(x, dtype=dtype)
        return self.cp.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        if isinstance(x, self.cp.ndarray):
            return x.get()
        elif self.sparse.issparse(x):
            return x.toarray().get()
        elif self.npsparse.issparse(x):
            return x.toarray()
        return np.array(x, copy=False)

    def issparse(self, x):
        return self.sparse.issparse(x) or self.npsparse.issparse(x)

    def zero_state(self, nqubits):
        n = 1 << nqubits
        kernel = self.gates.get(f"initial_state_kernel_{self.kernel_type}")
        state = self.cp.zeros(n, dtype=self.dtype)
        kernel((1,), (1,), [state])
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def zero_density_matrix(self, nqubits):
        n = 1 << nqubits
        kernel = self.gates.get(f"initial_state_kernel_{self.kernel_type}")
        state = self.cp.zeros(n * n, dtype=self.dtype)
        kernel((1,), (1,), [state])
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state.reshape((n, n))

    #def asmatrix_special(self, gate): Inherited from ``NumpyBackend``

    #def control_matrix(self, gate): Inherited from ``NumpyBackend``

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
        qubits = super()._create_qubits_tensor(gate, nqubits)
        return self.cp.asarray(qubits, dtype=self.cp.int32)

    def _as_custom_matrix(self, gate):
        matrix = super()._as_custom_matrix(gate)
        return self.cp.asarray(matrix.ravel())

    #def apply_gate(self, gate, state, nqubits): Inherited from ``NumbaBackend``
    
    #def apply_gate_density_matrix(self, gate, state, nqubits, inverse=False): Inherited from ``NumbaBackend``
    
    #def _apply_ygate_density_matrix(self, gate, state, nqubits): Inherited from ``NumbaBackend``

    #def apply_channel(self, gate): Inherited from ``NumbaBackend``

    #def apply_channel_density_matrix(self, channel, state, nqubits): Inherited from ``NumbaBackend``

    #def collapse_state(self, gate, state, nqubits):
    # TODO: Implement this

    #def collapse_density_matrix(self, gate, state, nqubits):
    # TODO: Implement this

    #def reset_error_density_matrix(self, gate, state, nqubits): Inherited from ``NumpyBackend``

    #def calculate_symbolic(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): Inherited from ``NumpyBackend``

    #def calculate_symbolic_density_matrix(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): Inherited from ``NumpyBackend``

    def calculate_probabilities(self, state, qubits, nqubits):
        try:
            probs = super().calculate_probabilities(state, qubits, nqubits)
        except MemoryError:
            # fall back to CPU
            probs = super().calculate_probabilities(self.to_numpy(state), qubits, nqubits)
        return probs

    def sample_shots(self, probabilities, nshots):
        # Sample shots on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_shots(probabilities, nshots)

    #def aggregate_shots(self, shots): Inherited from ``NumpyBackend``

    #def samples_to_binary(self, samples, nqubits): Inherited from ``NumpyBackend``

    #def samples_to_decimal(self, samples, nqubits): Inherited from ``NumpyBackend``

    def sample_frequencies(self, probabilities, nshots):
        # Sample frequencies on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_frequencies(probabilities, nshots)

    #def calculate_frequencies(self, samples): Inherited from ``NumpyBackend``

    #def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): Inherited from ``NumpyBackend``

    def calculate_expectation_state(self, matrix, state, normalize):
        state = self.cast(state)
        statec = self.cp.conj(state)
        hstate = matrix @ state
        ev = self.cp.real(self.cp.sum(statec * hstate))
        if normalize:
            norm = self.cp.sum(self.cp.square(self.cp.abs(state)))
            ev = ev / norm
        return ev

    def calculate_expectation_density_matrix(self, matrix, state, normalize):
        state = self.cast(state)
        ev = self.cp.real(self.cp.trace(matrix @ state))
        if normalize:
            norm = self.cp.real(self.cp.trace(state))
            ev = ev / norm
        return ev

    def calculate_eigenvalues(self, matrix, k=6):
        if self.issparse(matrix):
            log.warning("Calculating sparse matrix eigenvectors because "
                        "sparse modules do not provide ``eigvals`` method.")
            return self.calculate_eigenvectors(matrix, k=k)[0]
        return self.cp.linalg.eigvalsh(matrix)

    def calculate_eigenvectors(self, matrix, k=6):
        if self.issparse(matrix):
            if k < matrix.shape[0]:
                # Fallback to numpy because cupy's ``sparse.eigh`` does not support 'SA'
                from scipy.sparse.linalg import eigsh  # pylint: disable=import-error
                result = eigsh(matrix.get(), k=k, which='SA')
                return self.cast(result[0]), self.cast(result[1])
            matrix = matrix.toarray()
        if self.is_hip:
            # Fallback to numpy because eigh is not implemented in rocblas
            result = self.np.linalg.eigh(self.to_numpy(matrix))
            return self.cast(result[0]), self.cast(result[1])
        else:
            return self.cp.linalg.eigh(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            if self.issparse(matrix):
                from scipy.sparse.linalg import expm
            else:
                from scipy.linalg import expm
            return expm(-1j * a * matrix.get())
        else:
            expd = self.cp.diag(self.cp.exp(-1j * a * eigenvalues))
            ud = self.cp.transpose(self.cp.conj(eigenvectors))
            return self.cp.matmul(eigenvectors, self.cp.matmul(expd, ud))

    def calculate_matrix_product(self, hamiltonian, o):
        if isinstance(o, hamiltonian.__class__):
            new_matrix = hamiltonian.matrix.dot(o.matrix)
            return hamiltonian.__class__(hamiltonian.nqubits, new_matrix)

        if isinstance(o, self.tensor_types):
            rank = len(tuple(o.shape))
            o = self.cast(o)
            if rank == 1: # vector
                return hamiltonian.matrix.dot(o[:, np.newaxis])[:, 0]
            elif rank == 2: # matrix
                return hamiltonian.matrix.dot(o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))
