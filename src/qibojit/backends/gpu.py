"""Module defining the Cupy and CuQuantum backends."""

from typing import List, Tuple, Union

import numpy as np
from qibo.backends import Backend
from qibo.config import log, raise_error
from qibo.gates.abstract import ParametrizedGate
from qibo.gates.gates import Unitary
from qibo.gates.measurements import M
from qibo.gates.special import CallbackGate, FusedGate
from qibo.result import CircuitResult, QuantumState
from scipy import sparse

from qibojit.backends.cpu import NumbaBackend
from qibojit.backends.matrices import (
    CupyMatrices,
    CustomCuQuantumMatrices,
    CustomMatrices,
)
from qibojit.custom_operators import raw_kernels
from qibojit.custom_operators.ops import measure_frequencies


class CupyBackend(Backend):  # pragma: no cover
    # CI does not have GPUs

    DEFAULT_BLOCK_SIZE = 1024
    MAX_NUM_TARGETS = 7

    def __init__(self):
        super().__init__()

        import cupy as cp  # pylint: disable=import-error,import-outside-toplevel
        import cupy_backends  # pylint: disable=import-error,import-outside-toplevel
        import cupyx.scipy.sparse as cp_sparse  # pylint: disable=import-error,import-outside-toplevel

        self.engine = cp
        self.npsparse = sparse
        self.sparse = cp_sparse
        self.is_hip = cupy_backends.cuda.api.runtime.is_hip
        self.measure_frequencies_op = measure_frequencies
        # number of available GPUs (for multigpu)
        self.ngpus = self.engine.cuda.runtime.getDeviceCount()

        self.name = "qibojit"
        self.platform = "cupy"
        self.versions["cupy"] = self.engine.__version__

        self.supports_multigpu = True
        self.numeric_types += (
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.tensor_types = (np.ndarray, self.engine.ndarray)

        self.device = "/GPU:0"
        self.matrices = CupyMatrices(self.dtype)
        self.custom_matrices = CustomMatrices(self.dtype)
        self.custom_matrices._cast = self.matrices._cast

        try:
            if not self.engine.cuda.runtime.getDeviceCount():  # pragma: no cover
                raise RuntimeError(
                    "Cannot use ``cupy`` backend if GPU is not available."
                )
        except self.engine.cuda.runtime.CUDARuntimeError as exc:
            raise ImportError("Could not detect ``cupy`` compatible devices.") from exc

        self.kernels = (
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

        type_replacements = {
            "float32": "float",
            "float64": "double",
            "complex64": "thrust::complex<float>",
            "complex128": "thrust::complex<double>",
        }

        def kernel_loader(name, dtype):
            code = getattr(raw_kernels, name)
            code = code.replace("T", type_replacements[dtype])
            if name == "initial_state_kernel":
                body = (
                    "state[0] = 1;"
                    if dtype in ("float32", "float64")
                    else f"state[0] = {type_replacements[dtype]}(1, 0);"
                )
                code = code.replace("<BODY>", body)
            gate = self.engine.RawKernel(code, name, ("--std=c++11",))
            self.gates[f"{name}_{dtype}"] = gate

        for dtype, _ in type_replacements.items():
            for name in self.kernels:
                kernel_loader(f"{name}_kernel", dtype)
                kernel_loader(f"multicontrol_{name}_kernel", dtype)
            kernel_loader("collapse_state_kernel", dtype)
            kernel_loader("initial_state_kernel", dtype)

        # load multiqubit kernels
        name = "apply_multi_qubit_gate_kernel"
        for ntargets in range(3, self.MAX_NUM_TARGETS + 1):
            for dtype, ctype in type_replacements.items():
                code = getattr(raw_kernels, name)
                code = code.replace("T", ctype)
                code = code.replace("nsubstates", str(2**ntargets))
                code = code.replace("MAX_BLOCK_SIZE", str(self.DEFAULT_BLOCK_SIZE))
                gate = self.engine.RawKernel(code, name, ("--std=c++11",))
                self.gates[f"{name}_{dtype}_{ntargets}"] = gate

    def cast(self, array, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype

        if self.sparse.issparse(array):
            if dtype != array.dtype:
                return array.astype(dtype)

            return array

        if self.npsparse.issparse(array):
            class_ = getattr(self.sparse, array.__class__.__name__)

            return class_(array, dtype=dtype)

        if isinstance(array, self.engine.ndarray) and copy:
            return self.engine.copy(self.engine.asarray(array, dtype=dtype))

        return self.engine.asarray(array, dtype=dtype)

    def is_sparse(self, array):
        return self.sparse.issparse(array) or self.npsparse.issparse(array)

    def set_device(self, device):
        if "GPU" not in device:
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )
        self.device = device

    def set_seed(self, seed: int):
        np.random.seed(seed)
        self.engine.random.seed(seed)

    def set_threads(self, nthreads):
        import numba  # pylint: disable=import-outside-toplevel

        numba.set_num_threads(nthreads)
        self.nthreads = nthreads

    def to_numpy(self, array):
        if isinstance(array, self.engine.ndarray):
            return array.get()

        if isinstance(array, list):
            return self.engine.asarray(array).get()

        if self.sparse.issparse(array):
            return array.toarray().get()

        if self.npsparse.issparse(array):
            return array.toarray()

        return np.asarray(array)

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def eig(self, array, **kwargs) -> "ndarray":
        cp_version = self.versions["cupy"]
        cp_version = int(cp_version.split(".")[0])

        if cp_version <= 13:
            log.warning(
                "Falling back to CPU due to lack of native ``linalg.eig`` implementation in"
                + f"``cupy=={self.versions['cupy']}``."
            )

            eigvals, eigvecs = np.linalg.eig(self.to_numpy(array), **kwargs)
            eigvals = self.cast(eigvals, dtype=eigvals.dtype)
            eigvecs = self.cast(eigvecs, dtype=eigvecs.dtype)

            return eigvals, eigvecs

        return super().eig(array, **kwargs)

    def eigvals(self, array, **kwargs) -> "ndarray":
        cp_version = self.versions["cupy"]
        cp_version = int(cp_version.split(".")[0])

        if cp_version <= 13:
            log.warning(
                "Falling back to CPU due to lack of native ``linalg.eigvals`` implementation"
                f"in ``cupy=={self.versions['cupy']}``."
            )

            eigvals = np.linalg.eigvals(self.to_numpy(array), **kwargs)

            return self.cast(eigvals, dtype=eigvals.dtype)

        return super().eig(array, **kwargs)

    def expm(self, array) -> "ndarray":
        if self.is_sparse(array):
            from scipy.linalg import (  # pylint: disable=import-outside-toplevel
                expm,
            )

            log.warning(
                "Falling back to CPU due to lack of native ``cupyx.scipy.sparse.linalg.expm``"
                + f"implementation in ``cupy=={self.versions['cupy']}``."
            )
            array = self.to_numpy(array)
        else:
            from cupyx.scipy.linalg import (  # pylint: disable=import-error,C0415
                expm,
            )

        exp_matrix = expm(array)

        return self.cast(exp_matrix, dtype=exp_matrix.dtype)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def matrix_power(
        self,
        matrix,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype=None,
    ):
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if dtype is None:
            dtype = self.dtype

        if isinstance(power, int) and power >= 0.0:
            return self.engine.linalg.matrix_power(matrix, power)

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return self._negative_power_singular_matrix(
                    matrix,
                    power,
                    precision_singularity,
                )

        log.warning(
            "Falling back to CPU due to lack of native ``linalg.fractional_matrix_power``"
            + f"implementation in ``cupy=={self.versions['cupy']}``."
        )

        copied = self.to_numpy(matrix)
        copied = super().matrix_power(
            copied,
            power=power,
            precision_singularity=precision_singularity,
            dtype=dtype,
        )

        return self.cast(copied, dtype=copied.dtype)

    ########################################################################################
    ######## Methods related to the creation and manipulation of quantum objects    ########
    ########################################################################################

    def maximally_mixed_state(self, nqubits, dtype=None):
        if dtype is None:
            dtype = self.dtype

        n = 1 << nqubits
        state = self.identity(n, dtype=self.dtype)
        self.engine.cuda.stream.get_current_stream().synchronize()

        return state.reshape((n, n)) / 2**nqubits

    def zero_state(self, nqubits, density_matrix: bool = False, dtype=None):
        if dtype is None:
            dtype = self.dtype

        n = 1 << nqubits
        shape = n * n if density_matrix else n
        kernel = self.gates.get(f"initial_state_kernel_{self.dtype}")
        state = self.zeros(shape, dtype=dtype)
        kernel((1,), (1,), [state])
        self.engine.cuda.stream.get_current_stream().synchronize()

        return state.reshape((n, n)) if density_matrix else state

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def collapse_state(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
        density_matrix: bool = False,
    ):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        nblocks, block_size = self._calculate_blocks(nstates)

        state = self.cast(state, dtype=state.dtype)
        qubits = self.cast(
            [nqubits - q - 1 for q in reversed(qubits)], dtype=self.int32
        )
        args = [state, qubits, int(shot), ntargets]
        kernel = self.gates.get(f"collapse_state_kernel_{self.dtype}")
        kernel((nblocks,), (block_size,), args)
        self.engine.cuda.stream.get_current_stream().synchronize()

        if normalize:
            norm = self.sqrt(self.sum(self.abs(state) ** 2))
            state = state / norm

        return state

    def execute_distributed_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=1000,
    ):
        import joblib  # pylint: disable=import-outside-toplevel

        if not circuit.queues.queues:
            circuit.queues.set(circuit.queue)

        try:
            cpu_backend = NumbaBackend()
            cpu_backend.set_dtype(self.dtype)
            ops = MultiGpuOps(self, cpu_backend, circuit)

            if initial_state is None:
                # Generate pieces for |000...0> state
                pieces = [cpu_backend.zero_state(circuit.nlocal)]
                pieces.extend(
                    np.zeros(2**circuit.nlocal, dtype=self.dtype)
                    for _ in range(circuit.ndevices - 1)
                )
            elif isinstance(initial_state, (CircuitResult, QuantumState)):
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
                    f"Initial state type {type(initial_state)} is not supported by "
                    + "distributed circuits.",
                )
            for gate in circuit.queue:
                if isinstance(gate, M):
                    gate.result.backend = CupyBackend()
            special_gates = iter(circuit.queues.special_queue)
            for queues in circuit.queues.queues:
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
                    circuit._final_state = CircuitResult(  # pylint: disable=W0212
                        state, circuit.measurements, self, nshots=nshots
                    )
                    return circuit._final_state  # pylint: disable=W0212

                circuit._final_state = QuantumState(  # pylint: disable=W0212
                    state, self
                )
                return circuit._final_state  # pylint: disable=W0212

            if circuit.measurements:
                circuit._final_state = CircuitResult(  # pylint: disable=W0212
                    state, circuit.measurements, self, nshots=nshots
                )
                return circuit._final_state  # pylint: disable=W0212

            circuit._final_state = QuantumState(state, self)  # pylint: disable=W0212

            return circuit._final_state  # pylint: disable=W0212

        except self.oom_error:
            raise_error(
                RuntimeError,
                "State does not fit in memory during distributed "
                "execution. Please create a new circuit with "
                "different device configuration and try again.",
            )

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def calculate_probabilities(
        self, state, qubits, nqubits: int, density_matrix: bool = False
    ):
        try:
            probs = super().calculate_probabilities(
                state, qubits, nqubits, density_matrix
            )
        except MemoryError:
            # fall back to CPU
            probs = super().calculate_probabilities(
                self.to_numpy(state), qubits, nqubits, density_matrix
            )

        return probs

    def sample_shots(self, probabilities, nshots):
        # Sample shots on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_shots(probabilities, nshots)

    def sample_frequencies(self, probabilities, nshots):
        # Sample frequencies on CPU
        probabilities = self.to_numpy(probabilities)
        return super().sample_frequencies(probabilities, nshots)

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _as_custom_matrix(self, gate):
        if isinstance(gate, Unitary):
            matrix = gate.parameters[0]
            if isinstance(matrix, self.engine.ndarray):
                return matrix.ravel()

        name = gate.__class__.__name__
        _matrix = getattr(self.custom_matrices, name)

        if name == "FanOut":
            matrix = _matrix(*gate.init_args)
        elif isinstance(gate, ParametrizedGate):
            if name == "GeneralizedRBS":  # pragma: no cover
                # this is tested in qibo tests
                theta = gate.init_kwargs["theta"]
                phi = gate.init_kwargs["phi"]
                matrix = _matrix(gate.init_args[0], gate.init_args[1], theta, phi)
            else:
                matrix = _matrix(*gate.parameters)
        elif isinstance(gate, FusedGate):  # pragma: no cover
            matrix = self.matrix_fused(gate)
        else:
            matrix = (
                _matrix(2 ** len(gate.target_qubits)) if callable(_matrix) else _matrix
            )

        return self.cast(matrix.ravel(), dtype=matrix.dtype)

    def _calculate_blocks(self, nstates, block_size=DEFAULT_BLOCK_SIZE):
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

    def _create_qubits_tensor(self, gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return self.cast(sorted(qubits), dtype=self.int32)

    def _multi_qubit_base(self, state, nqubits, targets, gate, qubits):
        assert gate is not None
        if qubits is None:
            qubits = self.cast(
                sorted(nqubits - q - 1 for q in targets), dtype=self.int32
            )
        ntargets = len(targets)
        if ntargets > self.MAX_NUM_TARGETS:
            raise ValueError(
                f"Number of target qubits must be <= {self.MAX_NUM_TARGETS}"
                f" but is {ntargets}."
            )
        nactive = len(qubits)
        targets = self.cast(
            tuple(1 << (nqubits - t - 1) for t in targets[::-1]),
            dtype=self.int64,
        )
        nstates = 1 << (nqubits - nactive)
        nblocks, block_size = self._calculate_blocks(nstates)
        kernel = self.gates.get(
            f"apply_multi_qubit_gate_kernel_{self.dtype}_{ntargets}"
        )
        args = (state, gate, qubits, targets, ntargets, nactive)
        kernel((nblocks,), (block_size,), args)
        self.engine.cuda.stream.get_current_stream().synchronize()
        return state

    def _one_qubit_base(self, state, nqubits, target, kernel, gate, qubits):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)
        if kernel in ("apply_x", "apply_y", "apply_z"):
            args = (state, tk, m)
        else:
            args = (state, tk, m, gate)

        if ncontrols:
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{self.dtype}")
            args += (qubits, ncontrols + 1)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{self.dtype}")

        nblocks, block_size = self._calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.engine.cuda.stream.get_current_stream().synchronize()
        return state

    def _two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits):
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
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{self.dtype}")
            args += (qubits, ncontrols + 2)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{self.dtype}")

        nblocks, block_size = self._calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.engine.cuda.stream.get_current_stream().synchronize()
        return state


class CuQuantumBackend(CupyBackend):  # pragma: no cover
    # CI does not test for GPU

    def __init__(self):
        super().__init__()
        import cuquantum  # pylint: disable=import-error,import-outside-toplevel
        from cuquantum import custatevec as cusv  # pylint: disable=import-error,C0415

        self.cuquantum = cuquantum
        self.cusv = cusv
        self.platform = "cuquantum"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
        self.handle = self.cusv.create()
        self.custom_matrices = CustomCuQuantumMatrices(self.dtype)
        self.custom_matrices._cast = self.matrices._cast

    def __del__(self):
        if hasattr(self, "cusv"):
            self.cusv.destroy(self.handle)

    def set_dtype(self, dtype):
        if dtype in ("float32", "float64"):
            raise_error(
                NotImplementedError,
                "``CuQuantumBackend only supports data types ``complex64`` and ``complex128``.",
            )

        if dtype != self.dtype:
            super().set_dtype(dtype)
            if self.custom_matrices:
                self.custom_matrices = CustomCuQuantumMatrices(self.dtype)

    def get_cuda_type(self, dtype="complex64"):
        if dtype not in ("complex128", "complex64"):
            raise_error(
                NotImplementedError,
                "``CuQuantumBackend only supports data types ``complex64`` and ``complex128``.",
            )

        if dtype == "complex128":
            return (
                self.cuquantum.cudaDataType.CUDA_C_64F,
                self.cuquantum.ComputeType.COMPUTE_64F,
            )

        return (
            self.cuquantum.cudaDataType.CUDA_C_32F,
            self.cuquantum.ComputeType.COMPUTE_32F,
        )

    def _one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None):
        ntarget = 1
        target = nqubits - target - 1
        if qubits is not None:
            ncontrols = len(qubits) - 1
            controls = set(list(self.to_numpy(qubits))) ^ {target}
        else:
            ncontrols = 0
            controls = self.empty(0)
        adjoint = 0
        target = self.to_numpy([target])
        target = target.astype(np.int32)

        controls = self.to_numpy(list(controls))
        controls = controls.astype(np.int32)

        assert state.dtype == gate.dtype

        data_type, compute_type = self.get_cuda_type(state.dtype)
        if isinstance(gate, self.engine.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspace_size = self.cusv.apply_matrix_get_workspace_size(
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
        if workspace_size > 0:
            workspace = self.engine.cuda.memory.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        print(type(target))

        self.cusv.apply_matrix(
            handle=self.handle,
            sv=state.data.ptr,
            sv_data_type=data_type,
            n_index_bits=nqubits,
            matrix=gate_ptr,
            matrix_data_type=data_type,
            layout=self.cusv.MatrixLayout.ROW,
            adjoint=adjoint,
            targets=target.ctypes.data,
            n_targets=ntarget,
            controls=controls.ctypes.data,
            control_bit_values=0,
            n_controls=ncontrols,
            compute_type=compute_type,
            extra_workspace=workspace_ptr,
            extra_workspace_size_in_bytes=workspace_size,
        )

        return state

    def _two_qubit_base(
        self, state, nqubits, target1, target2, kernel, gate, qubits=None
    ):
        ntarget = 2
        target1 = nqubits - target1 - 1
        target2 = nqubits - target2 - 1
        target = np.asarray([target2, target1], dtype=self.int32)
        if qubits is not None:
            ncontrols = len(qubits) - 2
            qubits = self.to_numpy(qubits)
            controls = np.asarray(
                [i for i in qubits if i not in [target1, target2]], dtype=self.int32
            )
        else:
            ncontrols = 0
            controls = np.empty(0)

        adjoint = 0

        state = self.cast(state, dtype=self.dtype)
        gate = self.cast(gate, dtype=self.dtype)

        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)

        if kernel == "apply_swap":
            n_bit_swaps = 1
            bit_swaps = [(target1, target2)]
            mask_len = ncontrols
            mask_bit_string = self.ones(ncontrols)
            mask_ordering = controls

            self.cusv.swap_index_bits(
                self.handle,
                state.data.ptr,
                data_type,
                nqubits,
                bit_swaps,
                n_bit_swaps,
                mask_bit_string,
                mask_ordering,
                mask_len,
            )
            return state

        if isinstance(gate, self.engine.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspace_size = self.cusv.apply_matrix_get_workspace_size(
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
        if workspace_size > 0:
            workspace = self.engine.cuda.memory.alloc(workspace_size)
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
            workspace_size,
        )

        return state

    def _multi_qubit_base(self, state, nqubits, targets, gate, qubits=None):
        state = self.cast(state, dtype=self.dtype)
        ntarget = len(targets)
        if qubits is None:
            qubits = sorted(nqubits - q - 1 for q in targets)
        else:
            qubits = self.to_numpy(qubits)
        target = [nqubits - q - 1 for q in targets]
        target = np.asarray(target[::-1], dtype=self.int32)
        controls = np.asarray([i for i in qubits if i not in target], dtype=self.int32)
        ncontrols = len(controls)
        adjoint = 0
        gate = self.cast(gate, dtype=self.dtype)
        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)

        if isinstance(gate, self.engine.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspace_size = self.cusv.apply_matrix_get_workspace_size(
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
        if workspace_size > 0:
            workspace = self.engine.cuda.memory.alloc(workspace_size)
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
            workspace_size,
        )

        return state

    def collapse_state(
        self,
        state,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
        density_matrix: bool = False,
    ):
        state = self.cast(state, dtype=self.dtype)
        results = bin(int(shot)).replace("0b", "")
        results = list(map(int, "0" * (len(qubits) - len(results)) + results))[::-1]
        ntarget = 1
        qubits = np.asarray([nqubits - q - 1 for q in reversed(qubits)], dtype="int32")
        data_type, _ = self.get_cuda_type(state.dtype)

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
            norm = self.engine.sqrt(
                self.engine.sum(self.engine.square(self.engine.abs(state)))
            )
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
            with self.backend.engine.cuda.Device(device_id):
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
