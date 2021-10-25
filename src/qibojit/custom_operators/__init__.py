from qibo.backends.abstract import AbstractBackend
from qibo.backends.numpy import NumpyBackend
from qibo.abstractions.states import AbstractState
from qibo.config import raise_error
from qibojit.custom_operators.backends import NumbaBackend


class CupyCpuDevice:

    def __init__(self, K):
        self.K = K

    def __enter__(self, *args):
        self.K.set_engine("numba")

    def __exit__(self, *args):
        if self.K.gpu_devices: # pragma: no cover
            self.K.set_engine("cupy")


class JITCustomBackend(NumpyBackend):

    description = "Uses custom operators based on numba.jit for CPU and " \
                  "custom CUDA kernels loaded with cupy GPU."

    def __init__(self):
        super().__init__()
        self.is_custom = True
        self.name = "qibojit"
        self.engine = None # active engine
        self._numba_engine = NumbaBackend()
        self._cupy_engine = None

        try: # pragma: no cover
            from cupy import cuda # pylint: disable=E0401
            ngpu = cuda.runtime.getDeviceCount()
        except:
            ngpu = 0

        import os
        if "OMP_NUM_THREADS" in os.environ: # pragma: no cover
            self.set_threads(int(os.environ.get("OMP_NUM_THREADS")))
        if "NUMBA_NUM_THREADS" in os.environ: # pragma: no cover
            self.set_threads(int(os.environ.get("NUMBA_NUM_THREADS")))

        self.cpu_devices = ["/CPU:0"]
        self.gpu_devices = [f"/GPU:{i}" for i in range(ngpu)]
        if self.gpu_devices: # pragma: no cover
            # CI does not use GPUs
            self.default_device = self.gpu_devices[0]
            self.set_engine("cupy")
        elif self.cpu_devices:
            self.default_device = self.cpu_devices[0]
            self.set_engine("numba")
            self.set_threads(self.nthreads)
        self.cupy_cpu_device = CupyCpuDevice(self)

        # enable multi-GPU if no macos
        import sys
        if sys.platform != "darwin":
            self.supports_multigpu = True

    def set_engine(self, name): # pragma: no cover
        """Switcher between ``cupy`` for GPU and ``numba`` for CPU."""
        if name == "numba":
            import numpy as xp
            self.tensor_types = (xp.ndarray,)
            self.engine = self._numba_engine
        elif name == "cupy":
            import cupy as xp # pylint: disable=E0401
            self.tensor_types = (self.np.ndarray, xp.ndarray)
            if self.cupy_engine is None:
                from qibojit.custom_operators.backends import CupyBackend
                self.cupy_engine = CupyBackend()
            self.engine = self._cupy_engine
        else:
            raise_error(ValueError, "Unknown engine {}.".format(name))
        self.backend = xp
        self.numeric_types = (int, float, complex, xp.int32,
                              xp.int64, xp.float32, xp.float64,
                              xp.complex64, xp.complex128)
        self.native_types = (xp.ndarray,)
        self.Tensor = xp.ndarray
        self.random = xp.random
        self.newaxis = xp.newaxis
        if "GPU" in self.default_device: # pragma: no cover
            with self.device(self.default_device):
                self.matrices.allocate_matrices()
        else:
            self.matrices.allocate_matrices()

    def set_device(self, name):
        AbstractBackend.set_device(self, name)
        if "GPU" in name: # pragma: no cover
            self.set_engine("cupy")
        else:
            self.set_engine("numba")

    def set_threads(self, nthreads):
        AbstractBackend.set_threads(self, nthreads)
        import numba # pylint: disable=E0401
        numba.set_num_threads(nthreads)

    def to_numpy(self, x):
        if isinstance(x, self.np.ndarray):
            return x
        elif isinstance(x, AbstractState): # TODO: Remove this
            x = x.numpy()
        return self.engine.to_numpy(x)

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.engine.cast(x, dtype=dtype)

    def check_shape(self, shape):
        if self.engine.name == "cupy" and isinstance(shape, self.Tensor): # pragma: no cover
            shape = shape.get()
        return shape

    def reshape(self, x, shape):
        return super().reshape(x, self.check_shape(shape))

    def eye(self, shape, dtype='DTYPECPX'):
        return super().eye(self.check_shape(shape), dtype=dtype)

    def zeros(self, shape, dtype='DTYPECPX'):
        return super().zeros(self.check_shape(shape), dtype=dtype)

    def ones(self, shape, dtype='DTYPECPX'):
        return super().ones(self.check_shape(shape), dtype=dtype)

    def expm(self, x):
        if self.engine.name == "cupy": # pragma: no cover
            # Fallback to numpy because cupy does not have expm
            if isinstance(x, self.native_types):
                x = x.get()
            return self.backend.asarray(super().expm(x))
        return super().expm(x)

    def unique(self, x, return_counts=False):
        if self.engine.name == "cupy": # pragma: no cover
            if isinstance(x, self.native_types):
                x = x.get()
            # Uses numpy backend always
        return super().unique(x, return_counts)

    def gather(self, x, indices=None, condition=None, axis=0):
        if self.engine.name == "cupy": # pragma: no cover
            # Fallback to numpy because cupy does not support tuple indexing
            if isinstance(x, self.native_types):
                x = x.get()
            if isinstance(indices, self.native_types):
                indices = indices.get()
            if isinstance(condition, self.native_types):
                condition = condition.get()
            result = super().gather(x, indices, condition, axis)
            return self.backend.asarray(result)
        return super().gather(x, indices, condition, axis)

    def device(self, device_name):
        # assume tf naming convention '/GPU:0'
        if self.engine.name == "numba":
            return super().device(device_name)
        else: # pragma: no cover
            if "GPU" in device_name:
                device_id = int(device_name.split(":")[-1])
                return self.backend.cuda.Device(device_id % len(self.gpu_devices))
            else:
                return self.cupy_cpu_device

    def initial_state(self, nqubits, is_matrix=False):
        return self.engine.initial_state(nqubits, self.dtypes('DTYPECPX'),
                                         is_matrix=is_matrix)

    def sample_frequencies(self, probs, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probs, nshots)
        if not isinstance(probs, self.np.ndarray): # pragma: no cover
            # not covered because GitHub CI does not have GPU
            probs = probs.get()
        dtype = self._dtypes.get('DTYPEINT')
        seed = self.np.random.randint(0, int(1e8), dtype=dtype)
        nqubits = int(self.np.log2(tuple(probs.shape)[0]))
        frequencies = self.np.zeros(2 ** nqubits, dtype=dtype)
        # always fall back to numba CPU backend because for ops not implemented on GPU
        frequencies = self._numba_engine.measure_frequencies(
            frequencies, probs, nshots, nqubits, seed, self.nthreads)
        return frequencies

    def create_einsum_cache(self, qubits, nqubits, ncontrol=None): # pragma: no cover
        raise_error(NotImplementedError)

    def einsum_call(self, cache, state, matrix): # pragma: no cover
        raise_error(NotImplementedError)

    def create_gate_cache(self, gate):
        cache = self.GateCache()
        qubits = [gate.nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(gate.nqubits - q - 1 for q in gate.target_qubits)
        cache.qubits_tensor = self.cast(sorted(qubits), dtype="int32")
        if gate.density_matrix:
            cache.target_qubits_dm = [q + gate.nqubits for q in gate.target_qubits]
        return cache

    def _state_vector_call(self, gate, state):
        return gate.gate_op(state, gate.nqubits, gate.target_qubits,
                            gate.cache.qubits_tensor)

    def state_vector_matrix_call(self, gate, state):
        return gate.gate_op(state, gate.custom_op_matrix, gate.nqubits, gate.target_qubits,
                            gate.cache.qubits_tensor)

    def _density_matrix_call(self, gate, state):
        qubits = gate.cache.qubits_tensor + gate.nqubits
        shape = state.shape
        state = gate.gate_op(state.flatten(), 2 * gate.nqubits,
                             gate.target_qubits, qubits)
        state = gate.gate_op(state, 2 * gate.nqubits, gate.cache.target_qubits_dm,
                             gate.cache.qubits_tensor)
        return self.reshape(state, shape)

    def density_matrix_matrix_call(self, gate, state):
        qubits = gate.cache.qubits_tensor + gate.nqubits
        shape = state.shape
        state = gate.gate_op(state.flatten(), gate.custom_op_matrix, 2 * gate.nqubits,
                             gate.target_qubits, qubits)
        adjmatrix = self.conj(gate.custom_op_matrix)
        state = gate.gate_op(state, adjmatrix, 2 * gate.nqubits, gate.cache.target_qubits_dm,
                             gate.cache.qubits_tensor)
        return self.reshape(state, shape)

    def _density_matrix_half_call(self, gate, state):
        qubits = gate.cache.qubits_tensor + gate.nqubits
        shape = state.shape
        state = gate.gate_op(state.flatten(), 2 * gate.nqubits,
                             gate.target_qubits, qubits)
        return self.reshape(state, shape)

    def density_matrix_half_matrix_call(self, gate, state):
        qubits = gate.cache.qubits_tensor + gate.nqubits
        shape = state.shape
        state = gate.gate_op(state.flatten(), gate.custom_op_matrix, 2 * gate.nqubits,
                             gate.target_qubits, qubits)
        return self.reshape(state, shape)

    def _result_tensor(self, result):
        n = len(result)
        return int(sum(2 ** (n - i - 1) * r for i, r in enumerate(result)))

    def state_vector_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        return gate.gate_op(state, gate.cache.qubits_tensor, result, gate.nqubits, True)

    def density_matrix_collapse(self, gate, state, result):
        result = self._result_tensor(result)
        qubits = gate.cache.qubits_tensor + gate.nqubits
        shape = state.shape
        state = gate.gate_op(state.flatten(), qubits, result, 2 * gate.nqubits, False)
        state = gate.gate_op(state, gate.cache.qubits_tensor, result,
                             2 * gate.nqubits, False)
        state = self.reshape(state, shape)
        return state / self.trace(state)

    def on_cpu(self):
        return self.cupy_cpu_device

    def cpu_cast(self, x, dtype='DTYPECPX'):
        try:
            return x.get()
        except AttributeError:
            return super().cpu_cast(x, dtype)

    def transpose_state(self, pieces, state, nqubits, order):
        original_shape = state.shape
        state = state.ravel()
        # always fall back to numba CPU backend because for ops not implemented on GPU
        state = self._numba_engine.transpose_state(pieces, state, nqubits, order)
        return self.reshape(state, original_shape)

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        if self.engine.name == "cupy": # pragma: no cover
            if isinstance(value, self.backend.ndarray):
                value = value.get()
            if isinstance(target, self.backend.ndarray):
                target = target.get()
        self.np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def apply_gate(self, state, gate, nqubits, targets, qubits=None):
        return self.engine.one_qubit_base(state, nqubits, *targets, "apply_gate", qubits, gate)

    def apply_x(self, state, nqubits, targets, qubits=None):
        return self.engine.one_qubit_base(state, nqubits, *targets, "apply_x", qubits)

    def apply_y(self, state, nqubits, targets, qubits=None):
        return self.engine.one_qubit_base(state, nqubits, *targets, "apply_y", qubits)

    def apply_z(self, state, nqubits, targets, qubits=None):
        return self.engine.one_qubit_base(state, nqubits, *targets, "apply_z", qubits)

    def apply_z_pow(self, state, gate, nqubits, targets, qubits=None):
        return self.engine.one_qubit_base(state, nqubits, *targets, "apply_z_pow", qubits, gate)

    def apply_two_qubit_gate(self, state, gate, nqubits, targets, qubits=None):
        return self.engine.two_qubit_base(state, nqubits, *targets, "apply_two_qubit_gate",
                                          qubits, gate)

    def apply_swap(self, state, nqubits, targets, qubits=None):
        return self.engine.two_qubit_base(state, nqubits, *targets, "apply_swap", qubits)

    def apply_fsim(self, state, gate, nqubits, targets, qubits=None):
        return self.engine.two_qubit_base(state, nqubits, *targets,
                                          "apply_fsim", qubits, gate)

    def apply_multiqubit_gate(self, state, gate, nqubits, targets, qubits=None):
        # FIXME: fall back to numba temporarily until we implement this for GPU
        return self._numba_engine.multiqubit_base(state, nqubits, targets, qubits, gate)

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        return self.engine.collapse_state(state, qubits, result, nqubits, normalize)

    def swap_pieces(self, piece0, piece1, new_global, nlocal):
        # always fall back to numba CPU backend because for ops not implemented on GPU
        return self._numba_engine.swap_pieces(piece0, piece1, new_global, nlocal)
