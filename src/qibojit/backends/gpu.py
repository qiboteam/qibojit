from typing import Union

import numpy as np
from qibo.backends.numpy import NumpyBackend
from qibo.config import log, raise_error

from qibojit.backends.cpu import NumbaBackend
from qibojit.backends.matrices import CupyMatrices, CuQuantumMatrices, CustomMatrices


class CupyBackend(NumbaBackend):  # pragma: no cover
    # CI does not have GPUs

    DEFAULT_BLOCK_SIZE = 1024
    MAX_NUM_TARGETS = 7

    def __init__(self):
        NumpyBackend.__init__(self)

        import cupy as cp  # pylint: disable=import-error
        import cupy_backends  # pylint: disable=import-error

        self.name = "qibojit"
        self.platform = "cupy"
        self.versions["cupy"] = cp.__version__

        self.supports_multigpu = True
        self.numeric_types = (
            int,
            float,
            complex,
            cp.int32,
            cp.int64,
            cp.float32,
            cp.float64,
            cp.complex64,
            cp.complex128,
        )
        self.tensor_types = (np.ndarray, cp.ndarray)
        from scipy import sparse

        self.npsparse = sparse
        self.sparse = cp.sparse
        self.device = "/GPU:0"
        self.kernel_type = "double"
        self.matrices = CupyMatrices(self.dtype)
        self.custom_matrices = CustomMatrices(self.dtype)
        try:
            if not cp.cuda.runtime.getDeviceCount():  # pragma: no cover
                raise RuntimeError("Cannot use cupy backend if GPU is not available.")
        except cp.cuda.runtime.CUDARuntimeError:
            raise ImportError("Could not detect cupy compatible devices.")

        self.cp = cp
        self.is_hip = cupy_backends.cuda.api.runtime.is_hip
        self.KERNELS = (
            "apply_gate",
            "apply_x",
            "apply_y",
            "apply_z",
            "apply_z_pow",
            "apply_two_qubit_gate",
            "apply_fsim",
            "apply_swap",
        )

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
                code = code.replace("nsubstates", str(2**ntargets))
                code = code.replace("MAX_BLOCK_SIZE", str(self.DEFAULT_BLOCK_SIZE))
                gate = cp.RawKernel(code, name, ("--std=c++11",))
                self.gates[f"{name}_{ktype}_{ntargets}"] = gate

        # load numba op for measuring frequencies
        from qibojit.custom_operators.ops import measure_frequencies

        self.measure_frequencies_op = measure_frequencies

        # number of available GPUs (for multigpu)
        self.ngpus = cp.cuda.runtime.getDeviceCount()

    def set_precision(self, precision):
        super().set_precision(precision)
        if self.dtype == "complex128":
            self.kernel_type = "double"
        elif self.dtype == "complex64":
            self.kernel_type = "float"

    def set_device(self, device):
        if "GPU" not in device:
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )
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
        elif isinstance(x, self.cp.ndarray) and copy:
            return self.cp.copy(self.cp.asarray(x, dtype=dtype))
        else:
            return self.cp.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        if isinstance(x, self.cp.ndarray):
            return x.get()
        elif isinstance(x, list):
            return self.cp.asarray(x).get()
        elif self.sparse.issparse(x):
            return x.toarray().get()
        elif self.npsparse.issparse(x):
            return x.toarray()
        return np.array(x, copy=False)

    def is_sparse(self, x):
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

    def identity_density_matrix(self, nqubits, normalize: bool = True):
        n = 1 << nqubits
        state = self.cp.eye(n, dtype=self.dtype)
        self.cp.cuda.stream.get_current_stream().synchronize()
        if normalize:
            state /= 2**nqubits
        return state.reshape((n, n))

    def plus_state(self, nqubits):
        state = self.cp.ones(2**nqubits, dtype=self.dtype)
        state /= self.cp.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.cp.ones(2 * (2**nqubits,), dtype=self.dtype)
        state /= 2**nqubits
        return state

    def matrix_fused(self, gate):
        npmatrix = super().matrix_fused(gate)
        return self.cast(npmatrix, dtype=self.dtype)

    # def control_matrix(self, gate): Inherited from ``NumpyBackend``

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
        if qubits is None:
            qubits = self.cast(
                sorted(nqubits - q - 1 for q in targets), dtype=self.cp.int32
            )
        ntargets = len(targets)
        if ntargets > self.MAX_NUM_TARGETS:
            raise ValueError(
                f"Number of target qubits must be <= {self.MAX_NUM_TARGETS}"
                f" but is {ntargets}."
            )
        nactive = len(qubits)
        targets = self.cp.asarray(
            tuple(1 << (nqubits - t - 1) for t in targets[::-1]), dtype=self.cp.int64
        )
        nstates = 1 << (nqubits - nactive)
        nsubstates = 1 << ntargets
        nblocks, block_size = self.calculate_blocks(nstates)
        kernel = self.gates.get(
            f"apply_multi_qubit_gate_kernel_{self.kernel_type}_{ntargets}"
        )
        args = (state, gate, qubits, targets, ntargets, nactive)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def _create_qubits_tensor(self, gate, nqubits):
        qubits = super()._create_qubits_tensor(gate, nqubits)
        return self.cp.asarray(qubits, dtype=self.cp.int32)

    def _as_custom_matrix(self, gate):
        from qibo.gates import Unitary

        if isinstance(gate, Unitary):
            matrix = gate.parameters[0]
            if isinstance(matrix, self.cp.ndarray):
                return matrix.ravel()

        matrix = super()._as_custom_matrix(gate)
        return self.cp.asarray(matrix.ravel())

    # def apply_gate(self, gate, state, nqubits): Inherited from ``NumbaBackend``

    # def apply_gate_density_matrix(self, gate, state, nqubits, inverse=False): Inherited from ``NumbaBackend``

    # def _apply_ygate_density_matrix(self, gate, state, nqubits): Inherited from ``NumbaBackend``

    # def apply_channel(self, gate): Inherited from ``NumbaBackend``

    # def apply_channel_density_matrix(self, channel, state, nqubits): Inherited from ``NumbaBackend``

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        nblocks, block_size = self.calculate_blocks(nstates)

        state = self.cast(state)
        qubits = self.cast(
            [nqubits - q - 1 for q in reversed(qubits)], dtype=self.cp.int32
        )
        args = [state, qubits, int(shot), ntargets]
        kernel = self.gates.get(f"collapse_state_kernel_{self.kernel_type}")
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()

        if normalize:
            norm = self.cp.sqrt(self.cp.sum(self.cp.square(self.cp.abs(state))))
            state = state / norm
        return state

    # def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True): Inherited from ``NumbaBackend``

    # def reset_error_density_matrix(self, gate, state, nqubits): Inherited from ``NumpyBackend``

    def execute_distributed_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=1000,
    ):
        import joblib
        from qibo.gates import M
        from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState

        if not circuit.queues.queues:
            circuit.queues.set(circuit.queue)

        try:
            cpu_backend = NumbaBackend()
            cpu_backend.set_precision(self.precision)
            ops = MultiGpuOps(self, cpu_backend, circuit)

            if initial_state is None:
                # Generate pieces for |000...0> state
                pieces = [cpu_backend.zero_state(circuit.nlocal)]
                pieces.extend(
                    np.zeros(2**circuit.nlocal, dtype=self.dtype)
                    for _ in range(circuit.ndevices - 1)
                )
            elif isinstance(initial_state, CircuitResult) or isinstance(
                initial_state, QuantumState
            ):
                # TODO: Implement this
                if isinstance(initial_state.execution_result, list):
                    pieces = initial_state.execution_result
                else:
                    pieces = ops.to_pieces(initial_state.state())
            elif isinstance(initial_state, self.tensor_types):
                pieces = ops.to_pieces(initial_state)
            else:
                raise_error(
                    TypeError,
                    "Initial state type {} is not supported by "
                    "distributed circuits.".format(type(initial_state)),
                )
            for gate in circuit.queue:
                if isinstance(gate, M):
                    gate.result.backend = CupyBackend()
            special_gates = iter(circuit.queues.special_queue)
            for i, queues in enumerate(circuit.queues.queues):
                if queues:  # standard gate
                    config = circuit.queues.device_to_ids.items()
                    pool = joblib.Parallel(n_jobs=circuit.ndevices, prefer="threads")
                    pool(
                        joblib.delayed(ops.apply_gates)(pieces, queues, ids, device)
                        for device, ids in config
                    )

                else:  # special gate
                    gate = next(special_gates)
                    if isinstance(gate, tuple):  # SWAP global-local qubit
                        global_qubit, local_qubit = gate
                        pieces = ops.swap(pieces, global_qubit, local_qubit)
                    else:
                        pieces = ops.apply_special_gate(pieces, gate)

            for gate in special_gates:  # pragma: no cover
                pieces = ops.apply_special_gate(pieces, gate)

            state = ops.to_tensor(pieces)

            if circuit.has_unitary_channel:
                # here we necessarily have `density_matrix=True`, otherwise
                # execute_circuit_repeated would have been called
                if circuit.measurements:
                    circuit._final_state = CircuitResult(
                        state, circuit.measurements, self, nshots=nshots
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = QuantumState(state, self)
                    return circuit._final_state
            else:
                if circuit.measurements:
                    circuit._final_state = CircuitResult(
                        state, circuit.measurements, self, nshots=nshots
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = QuantumState(state, self)
                    return circuit._final_state

        except self.oom_error:
            raise_error(
                RuntimeError,
                "State does not fit in memory during distributed "
                "execution. Please create a new circuit with "
                "different device configuration and try again.",
            )

    # def calculate_symbolic(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): Inherited from ``NumpyBackend``

    # def calculate_symbolic_density_matrix(self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20): Inherited from ``NumpyBackend``

    def calculate_probabilities(self, state, qubits, nqubits):
        try:
            probs = super().calculate_probabilities(state, qubits, nqubits)
        except MemoryError:
            # fall back to CPU
            probs = super().calculate_probabilities(
                self.to_numpy(state), qubits, nqubits
            )
        return probs

    def sample_shots(self, probabilities, nshots):
        # Sample shots on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_shots(probabilities, nshots)

    # def aggregate_shots(self, shots): Inherited from ``NumpyBackend``

    # def samples_to_binary(self, samples, nqubits): Inherited from ``NumpyBackend``

    # def samples_to_decimal(self, samples, nqubits): Inherited from ``NumpyBackend``

    def sample_frequencies(self, probabilities, nshots):
        # Sample frequencies on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_frequencies(probabilities, nshots)

    # def calculate_frequencies(self, samples): Inherited from ``NumpyBackend``

    # def assert_allclose(self, value, target, rtol=1e-7, atol=0.0): Inherited from ``NumpyBackend``

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

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if not hermitian:
            log.warning(
                "Falling back to CPU for eigenvalue calculation of a non-Hermitian operator."
            )

            matrix_cpu = self.to_numpy(matrix)
            eigenvalues = super().calculate_eigenvalues(matrix_cpu, k, hermitian)

            return self.cast(eigenvalues, dtype=eigenvalues.dtype)

        if self.is_sparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.calculate_eigenvectors(matrix, k=k)[0]
        return self.cp.linalg.eigvalsh(matrix)

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: bool = True):
        if not hermitian:
            log.warning(
                "Falling back to CPU for eigenvector calculation of a non-Hermitian operator."
            )

            matrix_cpu = self.to_numpy(matrix)
            eigenvalues, eigenvectors = super().calculate_eigenvectors(
                matrix_cpu, k, hermitian
            )
            eigenvalues = self.cast(eigenvalues, dtype=eigenvalues.dtype)
            eigenvectors = self.cast(eigenvectors, dtype=eigenvectors.dtype)

            return eigenvalues, eigenvectors

        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                # Fallback to numpy because cupy's ``sparse.eigh`` does not support 'SA'
                from scipy.sparse.linalg import eigsh  # pylint: disable=import-error

                result = eigsh(matrix.get(), k=k, which="SA")
                return self.cast(result[0]), self.cast(result[1])
            matrix = matrix.toarray()
        if self.is_hip:
            # Fallback to numpy because eigh is not implemented in rocblas
            result = self.np.linalg.eigh(self.to_numpy(matrix))
            return self.cast(result[0]), self.cast(result[1])

        return self.cp.linalg.eigh(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            if self.is_sparse(matrix):
                from scipy.sparse.linalg import expm

                return self.cast(expm(-1j * a * self.to_numpy(matrix)))
            else:
                from cupyx.scipy.linalg import expm  # pylint: disable=import-error

                return self.cast(expm(-1j * a * matrix))
        else:
            expd = self.cp.diag(self.cp.exp(-1j * a * eigenvalues))
            ud = self.cp.transpose(self.cp.conj(eigenvectors))
            return self.cp.matmul(eigenvectors, self.cp.matmul(expd, ud))

    def calculate_matrix_power(self, matrix, power: Union[float, int]):
        if isinstance(power, int):
            return self.cp.linalg.matrix_power(matrix, power)

        copied = self.to_numpy(matrix)
        copied = super().calculate_matrix_power(copied, power)

        return self.cast(copied, dtype=copied.dtype)


class CuQuantumBackend(CupyBackend):  # pragma: no cover
    # CI does not test for GPU

    def __init__(self):
        super().__init__()
        import cuquantum  # pylint: disable=import-error
        from cuquantum import custatevec as cusv  # pylint: disable=import-error

        self.cuquantum = cuquantum
        self.cusv = cusv
        self.platform = "cuquantum"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
        self.handle = self.cusv.create()
        self.custom_matrices = CuQuantumMatrices(self.dtype)

    def __del__(self):
        if hasattr(self, "cusv"):
            self.cusv.destroy(self.handle)

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)
            if self.custom_matrices:
                self.custom_matrices = CuQuantumMatrices(self.dtype)

    def get_cuda_type(self, dtype="complex64"):
        if dtype == "complex128":
            return (
                self.cuquantum.cudaDataType.CUDA_C_64F,
                self.cuquantum.ComputeType.COMPUTE_64F,
            )
        elif dtype == "complex64":
            return (
                self.cuquantum.cudaDataType.CUDA_C_32F,
                self.cuquantum.ComputeType.COMPUTE_32F,
            )
        else:
            raise TypeError("Type can be either complex64 or complex128")

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None):
        ntarget = 1
        target = nqubits - target - 1
        if qubits is not None:
            qubits = self.to_numpy(qubits)
            ncontrols = len(qubits) - 1
            controls = self.np.asarray(
                [i for i in qubits if i != target], dtype="int32"
            )
        else:
            ncontrols = 0
            controls = self.np.empty(0)
        adjoint = 0
        target = self.np.asarray([target], dtype=self.np.int32)

        state = self.cast(state)
        gate = self.cast(gate)
        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)
        if isinstance(gate, self.cp.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, self.np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspaceSize = self.cusv.apply_matrix_get_workspace_size(
            self.handle,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            ntarget,
            ncontrols,
            compute_type,
        )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(
            self.handle,
            state.data.ptr,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            target.ctypes.data,
            ntarget,
            controls.ctypes.data,
            0,
            ncontrols,
            compute_type,
            workspace_ptr,
            workspaceSize,
        )

        return state

    def two_qubit_base(
        self, state, nqubits, target1, target2, kernel, gate, qubits=None
    ):
        ntarget = 2
        target1 = nqubits - target1 - 1
        target2 = nqubits - target2 - 1
        target = self.np.asarray([target2, target1], dtype=self.np.int32)
        if qubits is not None:
            ncontrols = len(qubits) - 2
            qubits = self.to_numpy(qubits)
            controls = self.np.asarray(
                [i for i in qubits if i not in [target1, target2]], dtype=self.np.int32
            )
        else:
            ncontrols = 0
            controls = self.np.empty(0)

        adjoint = 0

        state = self.cast(state)
        gate = self.cast(gate)

        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)

        if kernel == "apply_swap":
            nBitSwaps = 1
            bitSwaps = [(target1, target2)]
            maskLen = ncontrols
            maskBitString = self.np.ones(ncontrols)
            maskOrdering = controls

            self.cusv.swap_index_bits(
                self.handle,
                state.data.ptr,
                data_type,
                nqubits,
                bitSwaps,
                nBitSwaps,
                maskBitString,
                maskOrdering,
                maskLen,
            )
            return state

        if isinstance(gate, self.cp.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, self.np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspaceSize = self.cusv.apply_matrix_get_workspace_size(
            self.handle,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            ntarget,
            ncontrols,
            compute_type,
        )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(
            self.handle,
            state.data.ptr,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            target.ctypes.data,
            ntarget,
            controls.ctypes.data,
            0,
            ncontrols,
            compute_type,
            workspace_ptr,
            workspaceSize,
        )

        return state

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits=None):
        state = self.cast(state)
        ntarget = len(targets)
        if qubits is None:
            qubits = sorted(nqubits - q - 1 for q in targets)
        else:
            qubits = self.to_numpy(qubits)
        target = [nqubits - q - 1 for q in targets]
        target = self.np.asarray(target[::-1], dtype=self.np.int32)
        controls = self.np.asarray(
            [i for i in qubits if i not in target], dtype=self.np.int32
        )
        ncontrols = len(controls)
        adjoint = 0
        gate = self.cast(gate)
        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)

        if isinstance(gate, self.cp.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, self.np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspaceSize = self.cusv.apply_matrix_get_workspace_size(
            self.handle,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            ntarget,
            ncontrols,
            compute_type,
        )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(
            self.handle,
            state.data.ptr,
            data_type,
            nqubits,
            gate_ptr,
            data_type,
            self.cusv.MatrixLayout.ROW,
            adjoint,
            target.ctypes.data,
            ntarget,
            controls.ctypes.data,
            0,
            ncontrols,
            compute_type,
            workspace_ptr,
            workspaceSize,
        )

        return state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        results = bin(int(shot)).replace("0b", "")
        results = list(map(int, "0" * (len(qubits) - len(results)) + results))[::-1]
        ntarget = 1
        qubits = self.np.asarray(
            [nqubits - q - 1 for q in reversed(qubits)], dtype="int32"
        )
        data_type, compute_type = self.get_cuda_type(state.dtype)

        for i in range(len(results)):
            self.cusv.collapse_on_z_basis(
                self.handle,
                state.data.ptr,
                data_type,
                nqubits,
                results[i],
                [qubits[i]],
                ntarget,
                1,
            )

        if normalize:
            norm = self.cp.sqrt(self.cp.sum(self.cp.square(self.cp.abs(state))))
            state = state / norm

        return state


class MultiGpuOps:  # pragma: no cover
    # CI does not have GPUs

    def __init__(self, backend, cpu_backend, circuit):
        self.backend = backend
        self.circuit = circuit
        self.cpu_ops = cpu_backend.ops

    def transpose_state(self, pieces, state, nqubits, order):
        original_shape = state.shape
        state = state.ravel()
        # always fall back to numba CPU backend because for ops not implemented on GPU
        state = self.cpu_ops.transpose_state(tuple(pieces), state, nqubits, order)
        return np.reshape(state, original_shape)

    def to_pieces(self, state):
        nqubits = self.circuit.nqubits
        qubits = self.circuit.queues.qubits
        shape = (self.circuit.ndevices, 2**self.circuit.nlocal)
        state = np.reshape(self.backend.to_numpy(state), shape)
        pieces = [state[i] for i in range(self.circuit.ndevices)]
        new_tensor = np.zeros(shape, dtype=state.dtype)
        new_tensor = self.transpose_state(
            pieces, new_tensor, nqubits, qubits.transpose_order
        )
        for i in range(self.circuit.ndevices):
            pieces[i] = new_tensor[i]
        return pieces

    def to_tensor(self, pieces):
        nqubits = self.circuit.nqubits
        qubits = self.circuit.queues.qubits
        if qubits.list == list(range(self.circuit.nglobal)):
            tensor = np.concatenate([x[np.newaxis] for x in pieces], axis=0)
            tensor = np.reshape(tensor, (2**nqubits,))
        elif qubits.list == list(range(self.circuit.nlocal, self.circuit.nqubits)):
            tensor = np.concatenate([x[:, np.newaxis] for x in pieces], axis=1)
            tensor = np.reshape(tensor, (2**nqubits,))
        else:  # fall back to the transpose op
            tensor = np.zeros(2**nqubits, dtype=self.backend.dtype)
            tensor = self.transpose_state(
                pieces, tensor, nqubits, qubits.reverse_transpose_order
            )
        return tensor

    def apply_gates(self, pieces, queues, ids, device):
        """Method that is parallelized using ``joblib``."""
        for i in ids:
            device_id = int(device.split(":")[-1]) % self.backend.ngpus
            with self.backend.cp.cuda.Device(device_id):
                piece = self.backend.cast(pieces[i])
                for gate in queues[i]:
                    piece = self.backend.apply_gate(gate, piece, self.circuit.nlocal)
            pieces[i] = self.backend.to_numpy(piece)
            del piece

    def apply_special_gate(self, pieces, gate):
        """Executes special gates on CPU.

        Currently special gates are ``Flatten`` or ``CallbackGate``.
        This method calculates the full state vector because special gates
        are not implemented for state pieces.
        """
        from qibo.gates import CallbackGate

        # Reverse all global SWAPs that happened so far
        pieces = self.revert_swaps(pieces, reversed(gate.swap_reset))
        state = self.to_tensor(pieces)
        if isinstance(gate, CallbackGate):
            gate.apply(self.backend, state, self.circuit.nqubits)
        else:
            state = gate.apply(self.backend, state, self.circuit.nqubits)
            pieces = self.to_pieces(state)
        # Redo all global SWAPs that happened so far
        pieces = self.revert_swaps(pieces, gate.swap_reset)
        return pieces

    def swap(self, pieces, global_qubit, local_qubit):
        m = self.circuit.queues.qubits.reduced_global.get(global_qubit)
        m = self.circuit.nglobal - m - 1
        t = 1 << m
        for g in range(self.circuit.ndevices // 2):
            i = ((g >> m) << (m + 1)) + (g & (t - 1))
            local_eff = self.circuit.queues.qubits.reduced_local.get(local_qubit)
            self.cpu_ops.swap_pieces(
                pieces[i], pieces[i + t], local_eff, self.circuit.nlocal
            )
        return pieces

    def revert_swaps(self, pieces, swap_pairs):
        for q1, q2 in swap_pairs:
            if q1 not in self.circuit.queues.qubits.set:
                q1, q2 = q2, q1
            pieces = self.swap(pieces, q1, q2)
        return pieces
