from abc import ABC, abstractmethod


class AbstractPlatform(ABC):

    def __init__(self): # pragma: no cover
        self.name = "abstract"
        self.gates = None
        self.ops = None
        self.test_regressions = {}
        self.supports_multigpu = False

    @abstractmethod
    def cast(self, x, dtype=None, order=None): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits=None): # pragma: no cover
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


class NumbaPlatform(AbstractPlatform):

    def __init__(self):
        # check if cache exists
        from pathlib import Path
        if not list((Path(__file__).parent / "__pycache__").glob("*.nbi")): # pragma: no cover
            from qibo.config import log
            log.info("Compiling kernels because qibojit is imported for the first time," \
                     "please wait. Compilation happens only once after installing qibojit.")

        import numpy as np
        from qibojit.custom_operators import gates, ops
        super().__init__()
        self.name = "numba"
        self.gates = gates
        self.ops = ops
        self.np = np
        self.multi_qubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel
            }

    def cast(self, x, dtype=None, order='K'):
        if isinstance(x, self.np.ndarray):
            if dtype is None:
                return x
            else:
                return x.astype(dtype, copy=False, order=order)
        else:
            try:
                x = self.np.array(x, dtype=dtype, order=order)
            # only for CuPy arrays, as implicit conversion raises TypeError
            # and you need to cast manually using x.get()
            except TypeError: # pragma: no cover
                x = self.np.array(x.get(), dtype=dtype, copy=False, order=order)
            return x

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        nstates = 1 << (nqubits - ncontrols - 1)
        if ncontrols:
            kernel = getattr(self.gates, "multicontrol_{}_kernel".format(kernel))
            return kernel(state, gate, qubits, nstates, m)
        kernel = getattr(self.gates, "{}_kernel".format(kernel))
        return kernel(state, gate, nstates, m)

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits=None):
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

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits=None):
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


class CupyPlatform(AbstractPlatform): # pragma: no cover
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

        super().__init__()
        self.name = "cupy"
        self.np = np
        self.cp = cp
        self.supports_multigpu = True
        self.is_hip = cupy_backends.cuda.api.runtime.is_hip
        self.KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
                        "apply_two_qubit_gate", "apply_fsim", "apply_swap")
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

    def cast(self, x, dtype=None, order='C'):
        if isinstance(x, self.cp.ndarray):
            if dtype is None:
                return x
            else:
                return x.astype(dtype, copy=False, order=order)
        return self.cp.asarray(x, dtype=dtype, order=order)

    def get_kernel_type(self, state):
        if state.dtype == self.cp.complex128:
            return "double"
        elif state.dtype == self.cp.complex64:
            return "float"
        raise TypeError("State of invalid type {}.".format(state.dtype))

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)
        state = self.cast(state)
        if kernel in ("apply_x", "apply_y", "apply_z"):
            args = (state, tk, m)
        else:
            args = (state, tk, m, self.cast(gate, dtype=state.dtype).flatten())

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{ktype}")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 1)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{ktype}")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits=None):
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
        if kernel == "apply_swap":
            args = (state, tk1, tk2, m1, m2, uk1, uk2)
        else:
            args = (state, tk1, tk2, m1, m2, uk1, uk2, self.cast(gate).flatten())
            assert state.dtype == args[-1].dtype

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get(f"multicontrol_{kernel}_kernel_{ktype}")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 2)
        else:
            kernel = self.gates.get(f"{kernel}_kernel_{ktype}")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits=None):
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
        kernel = self.gates.get(f"apply_multi_qubit_gate_kernel_{ktype}_{ntargets}")
        args = (state, gate, qubits, targets, ntargets, nactive)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def initial_state(self, nqubits, dtype, is_matrix=False):
        n = 1 << nqubits
        if dtype in {"complex128", self.np.complex128, self.cp.complex128}:
            ktype = "double"
        elif dtype in {"complex64", self.np.complex64, self.cp.complex64}:
            ktype = "float"
        else: # pragma: no cover
            raise TypeError("Unknown dtype {} passed in initial state operator."
                            "".format(dtype))
        kernel = self.gates.get(f"initial_state_kernel_{ktype}")

        if is_matrix:
            state = self.cp.zeros(n * n, dtype=dtype)
            kernel((1,), (1,), [state])
            self.cp.cuda.stream.get_current_stream().synchronize()
            state = state.reshape((n, n))
        else:
            state = self.cp.zeros(n, dtype=dtype)
            kernel((1,), (1,), [state])
            self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        nblocks, block_size = self.calculate_blocks(nstates)

        state = self.cast(state)
        ktype = self.get_kernel_type(state)
        args = [state, self.cast(qubits, dtype=self.cp.int32), result, ntargets]
        kernel = self.gates.get(f"collapse_state_kernel_{ktype}")
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()

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


class CuQuantumPlatform(CupyPlatform): # pragma: no cover
    # CI does not test for GPU

    def __init__(self):
        super().__init__()
        import cuquantum # pylint: disable=import-error
        from cuquantum import custatevec as cusv # pylint: disable=import-error
        self.cuquantum = cuquantum
        self.cusv = cusv
        self.name = "cuquantum"
        self.supports_multigpu = False
        self.handle = self.cusv.create()

    def __del__(self):
        self.cusv.destroy(self.handle)
        super().__del__() # pylint: disable=E1101

    def get_cuda_type(self, dtype='complex64'):
        if dtype == 'complex128':
            return self.cuquantum.cudaDataType.CUDA_C_64F, self.cuquantum.ComputeType.COMPUTE_64F
        elif dtype == 'complex64':
            return self.cuquantum.cudaDataType.CUDA_C_32F, self.cuquantum.ComputeType.COMPUTE_32F
        else:
            raise TypeError("Type can be either complex64 or complex128")

    def one_qubit_base(self, state, nqubits, target, kernel, gate, qubits=None):
        ntarget = 1
        target = nqubits - target - 1
        target = self.np.asarray([target], dtype = self.np.int32)
        if qubits is not None:
            ncontrols = len(qubits) - 1
            controls = self.np.asarray([i for i in qubits if i != target], dtype = self.np.int32)
        else:
            ncontrols = 0
            controls = self.np.empty(0)
        adjoint = 0

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

        workspaceSize = self.cusv.apply_matrix_buffer_size(self.handle,
                                                           data_type,
                                                           nqubits,
                                                           gate_ptr,
                                                           data_type,
                                                           self.cusv.MatrixLayout.ROW,
                                                           adjoint,
                                                           ntarget,
                                                           ncontrols,
                                                           compute_type
                                                           )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(self.handle,
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
                               ncontrols,
                               0,
                               compute_type,
                               workspace_ptr,
                               workspaceSize
                               )

        return state

    def two_qubit_base(self, state, nqubits, target1, target2, kernel, gate, qubits=None):
        ntarget = 2
        target1 = nqubits - target1 - 1
        target2 = nqubits - target2 - 1
        target = self.np.asarray([target2, target1], dtype=self.np.int32)
        if qubits is not None:
            ncontrols = len(qubits) - 2
            controls = self.np.asarray([i for i in qubits if i not in [target1, target2]], dtype = self.np.int32)
        else:
            ncontrols = 0
            controls = self.np.empty(0)

        adjoint = 0

        state = self.cast(state)
        gate = self.cast(gate)

        assert state.dtype == gate.dtype
        data_type, compute_type = self.get_cuda_type(state.dtype)

        if kernel == 'apply_swap' and ncontrols == 0:
            nBasisBits = 2
            maskLen = 0
            maskBitString = 0
            maskOrdering = 0
            basisBits = target
            permutation  = self.np.asarray([0, 2, 1, 3], dtype=self.np.int64)
            diagonals  = self.np.asarray([1, 1, 1, 1], dtype=state.dtype)

            workspaceSize = self.cusv.apply_generalized_permutation_matrix_buffer_size(
                self.handle,
                data_type,
                nqubits,
                permutation.ctypes.data,
                diagonals.ctypes.data,
                data_type,
                basisBits,
                nBasisBits,
                maskLen)

            if workspaceSize > 0:
                workspace = self.cp.cuda.memory.alloc(workspaceSize)
                workspace_ptr = workspace.ptr
            else:
                workspace_ptr = 0

            # apply matrix
            self.cusv.apply_generalized_permutation_matrix(
                self.handle, state.data.ptr, data_type, nqubits,
                permutation.ctypes.data, diagonals.ctypes.data, data_type, adjoint,
                basisBits, nBasisBits, maskBitString, maskOrdering, maskLen,
                workspace_ptr, workspaceSize)

            return state

        if isinstance(gate, self.cp.ndarray):
            gate_ptr = gate.data.ptr
        elif isinstance(gate, self.np.ndarray):
            gate_ptr = gate.ctypes.data
        else:
            raise ValueError

        workspaceSize = self.cusv.apply_matrix_buffer_size(self.handle,
                                                           data_type,
                                                           nqubits,
                                                           gate_ptr,
                                                           data_type,
                                                           self.cusv.MatrixLayout.ROW,
                                                           adjoint,
                                                           ntarget,
                                                           ncontrols,
                                                           compute_type
                                                           )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(self.handle,
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
                               ncontrols,
                               0,
                               compute_type,
                               workspace_ptr,
                               workspaceSize
                               )

        return state

    def multi_qubit_base(self, state, nqubits, targets, gate, qubits=None):
        state = self.cast(state)
        ntarget = len(targets)
        if qubits is None:
            qubits = self.cast(sorted(nqubits - q - 1 for q in targets), dtype = self.cp.int32)
        target = [nqubits - q - 1 for q in targets]
        target = self.np.asarray(target[::-1], dtype = self.np.int32)
        controls = self.np.asarray([i for i in qubits if i not in target], dtype = self.np.int32)
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

        workspaceSize = self.cusv.apply_matrix_buffer_size(self.handle,
                                                           data_type,
                                                           nqubits,
                                                           gate_ptr,
                                                           data_type,
                                                           self.cusv.MatrixLayout.ROW,
                                                           adjoint,
                                                           ntarget,
                                                           ncontrols,
                                                           compute_type
                                                           )

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = self.cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        self.cusv.apply_matrix(self.handle,
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
                               ncontrols,
                               0,
                               compute_type,
                               workspace_ptr,
                               workspaceSize
                               )

        return state

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        state = self.cast(state)
        results = bin(result).replace("0b", "")
        results = list(map(int,  '0'* (len(qubits) - len(results)) + results))[::-1]
        ntarget = 1
        qubits = self.np.asarray(qubits, dtype = self.np.int32)
        data_type, compute_type = self.get_cuda_type(state.dtype)

        for i  in range(len(results)):
            self.cusv.collapse_on_z_basis(self.handle,
                                          state.data.ptr,
                                          data_type,
                                          nqubits,
                                          results[i],
                                          [qubits[i]],
                                          ntarget,
                                          1
                                          )

        if normalize:
            norm  = self.cp.sqrt(self.cp.sum(self.cp.square(self.cp.abs(state))))
            state = state / norm

        return state
