"""Module defining the Numba backend."""

import sys
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from qibo.backends import Backend, NumpyMatrices
from qibo.config import SHOT_METROPOLIS_THRESHOLD, raise_error
from qibo.gates.abstract import Gate, ParametrizedGate
from qibo.gates.special import FusedGate
from scipy.linalg import block_diag, expm, fractional_matrix_power, logm
from scipy.sparse import csr_matrix
from scipy.sparse import eye as eye_sparse
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import expm as expm_sparse

from qibojit.backends.matrices import CustomMatrices
from qibojit.custom_operators.quantum_info import QINFO

GATE_OPS = {
    "X": "apply_x",
    "CNOT": "apply_x",
    "TOFFOLI": "apply_x",
    "Y": "apply_y",
    "Z": "apply_z",
    "CY": "apply_y",
    "CZ": "apply_z",
    "U1": "apply_z_pow",
    "CU1": "apply_z_pow",
    "SWAP": "apply_swap",
    "fSim": "apply_fsim",
    "GeneralizedfSim": "apply_fsim",
}


class NumbaBackend(Backend):
    def __init__(self):
        super().__init__()

        import numba  # pylint: disable=import-outside-toplevel
        import psutil  # pylint: disable=import-outside-toplevel

        # avoiding circular imports
        from qibojit import __version__ as qibojit_version  # pylint: disable=C0415
        from qibojit.custom_operators import (  # pylint: disable=import-outside-toplevel
            gates,
            ops,
        )

        self.engine = np
        self.ops = ops

        self.tensor_types = (self.engine.ndarray,)
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )

        self.matrices = NumpyMatrices(self.dtype)
        self.custom_matrices = CustomMatrices(self.dtype)
        self.device = "/CPU:0"
        self.gates = gates
        self.name = "qibojit"
        self.measure_frequencies_op = self.ops.measure_frequencies
        self.multi_qubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel,
        }
        self.platform = "numba"
        self.versions.update(
            {
                "qibojit": qibojit_version,
                "numba": numba.__version__,
            }
        )

        if sys.platform == "darwin":  # pragma: no cover
            self.set_threads(psutil.cpu_count(logical=False))
        else:
            self.set_threads(len(psutil.Process().cpu_affinity()))

        # load the quantum info custom operators
        for method in dir(QINFO):
            if method[:2] != "__":
                setattr(self.qinfo, method, getattr(QINFO, method))

    def cast(
        self, array: ArrayLike, dtype: Optional[DTypeLike] = None, copy: bool = False
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype

        if isinstance(array, self.tensor_types):
            return array.astype(dtype, copy=copy)

        if self.is_sparse(array):
            return array.astype(dtype, copy=copy)

        return self.engine.asarray(array, dtype=dtype, copy=copy if copy else None)

    def is_sparse(self, array: ArrayLike) -> bool:
        """Determine if a given array is a sparse tensor."""
        return issparse(array)

    def set_device(self, device: str) -> None:
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_dtype(self, dtype: DTypeLike) -> None:
        if dtype != self.dtype:
            super().set_dtype(dtype)
            self.matrices = NumpyMatrices(self.dtype)
            if self.custom_matrices:
                self.custom_matrices = CustomMatrices(self.dtype)

    def set_seed(self, seed: Union[int, None]) -> None:
        super().set_seed(seed)
        if seed is not None:
            self.ops.set_seed(seed)
            self.qinfo.set_seed(seed)

    def set_threads(self, nthreads: int) -> None:
        import numba  # pylint: disable=import-outside-toplevel

        numba.set_num_threads(nthreads)
        self.nthreads = nthreads

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        if self.is_sparse(array):
            return array.toarray()
        return array

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def block_diag(self, *arrays: ArrayLike) -> ArrayLike:  # pragma: no cover
        return block_diag(*arrays)

    def csr_matrix(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return csr_matrix(array, **kwargs)

    def eigsh(self, array: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        return eigsh(array, **kwargs)

    def expm(self, array: ArrayLike) -> ArrayLike:
        func = expm_sparse if self.is_sparse(array) else expm
        return func(array)

    def logm(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        return logm(array, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def matrix_power(
        self,
        matrix: ArrayLike,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:  # pragma: no cover
        """Calculate the (fractional) ``power`` :math:`\\alpha` of ``matrix`` :math:`A`,
        i.e. :math:`A^{\\alpha}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU whenever ``power`` is not
            an integer.
        """
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if dtype is None:
            dtype = self.dtype

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return self._negative_power_singular_matrix(
                    matrix, power, precision_singularity, dtype=dtype
                )

        return fractional_matrix_power(matrix, power)

    ########################################################################################
    ######## Methods related to the creation and manipulation of quantum objects    ########
    ########################################################################################

    def zero_state(
        self,
        nqubits: int,
        density_matrix: bool = False,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype

        dims = 2**nqubits
        shape = 2 * (dims,) if density_matrix else dims
        state = self.empty(shape, dtype=dtype)

        func = (
            self.ops.initial_density_matrix
            if density_matrix
            else self.ops.initial_state_vector
        )

        return func(state)

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def apply_channel(
        self, channel: "Channel", state: ArrayLike, nqubits: int  # type: ignore
    ) -> ArrayLike:
        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            return self._apply_channel_density_matrix(channel, state, nqubits)

        return super().apply_channel(channel, state, nqubits)  # pragma: no cover

    def apply_gate(
        self, gate: Gate, state: ArrayLike, nqubits: int, inverse: bool = False
    ) -> ArrayLike:
        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            return self._apply_gate_density_matrix(gate, state, nqubits, inverse)

        return self._apply_gate(gate, state, nqubits)

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def sample_frequencies(self, probabilities: ArrayLike, nshots: int) -> Counter:
        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probabilities, nshots)

        seed = self.random_integers(0, int(1e8), dtype=self.int64)[0]
        nqubits = int(self.log2(tuple(probabilities.shape)[0]))
        frequencies = self.zeros(2**nqubits, dtype=self.int64)
        # always fall back to numba CPU backend because for ops not implemented on GPU
        frequencies = self.measure_frequencies_op(
            frequencies, probabilities, nshots, nqubits, seed, self.nthreads
        )
        return Counter({i: f for i, f in enumerate(frequencies) if f > 0})

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _apply_channel_density_matrix(
        self, channel: "Channel", state: ArrayLike, nqubits: int  # type: ignore
    ) -> ArrayLike:
        state = self.cast(state, dtype=state.dtype)
        if not channel._all_unitary_operators:  # pylint: disable=protected-access
            state_copy = self.cast(state, copy=True)
        new_state = (1 - channel.coefficient_sum) * state
        for coeff, gate in zip(channel.coefficients, channel.gates):
            state = self.apply_gate(gate, state, nqubits)
            new_state += coeff * state
            # reset the state
            if not channel._all_unitary_operators:  # pylint: disable=protected-access
                state = self.cast(state_copy, copy=True)
            else:
                state = self.apply_gate(gate, state, nqubits, inverse=True)
        return new_state

    def _apply_fanout_gate(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
        from qibo import gates  # pylint: disable=import-outside-toplevel

        control, targets = gate.control_qubits[0], gate.target_qubits

        for target in targets:
            gate = gates.CNOT(control, target)
            matrix = self._as_custom_matrix(gate)
            qubits = self._create_qubits_tensor(gate, nqubits)
            op = GATE_OPS.get("CNOT")
            state = self._one_qubit_base(state, nqubits, target, op, matrix, qubits)

        return state

    def _apply_gate(self, gate: Gate, state: ArrayLike, nqubits: int) -> ArrayLike:
        matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        targets = gate.target_qubits
        state = self.cast(state, dtype=state.dtype)

        if gate.name == "fanout":
            return self._apply_fanout_gate(gate, state, nqubits)

        if len(targets) == 1:
            op = GATE_OPS.get(gate.__class__.__name__, "apply_gate")
            return self._one_qubit_base(state, nqubits, *targets, op, matrix, qubits)

        if len(targets) == 2:
            op = GATE_OPS.get(gate.__class__.__name__, "apply_two_qubit_gate")
            return self._two_qubit_base(state, nqubits, *targets, op, matrix, qubits)

        return self._multi_qubit_base(state, nqubits, targets, matrix, qubits)

    def _apply_gate_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int, inverse: bool = False
    ) -> ArrayLike:
        name = gate.__class__.__name__
        if name in ["Y", "CY"]:
            return self._apply_ygate_density_matrix(gate, state, nqubits)

        if inverse:
            # used to reset the state when applying channels
            # see :meth:`qibojit.backend.NumpyBackend.apply_channel_density_matrix` below
            matrix = self.inv(gate.matrix(self))
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
            state = self._one_qubit_base(
                state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm
            )
            state = self._one_qubit_base(
                state, 2 * nqubits, *targets_dm, op, self.conj(matrix), qubits
            )
        elif len(targets) == 2:
            op = GATE_OPS.get(name, "apply_two_qubit_gate")
            state = self._two_qubit_base(
                state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm
            )
            state = self._two_qubit_base(
                state, 2 * nqubits, *targets_dm, op, self.conj(matrix), qubits
            )
        else:
            state = self._multi_qubit_base(
                state.ravel(), 2 * nqubits, targets, matrix, qubits_dm
            )
            state = self._multi_qubit_base(
                state, 2 * nqubits, targets_dm, self.conj(matrix), qubits
            )

        return self.reshape(state, shape)

    def _apply_ygate_density_matrix(
        self, gate: Gate, state: ArrayLike, nqubits: int
    ) -> ArrayLike:
        matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        qubits_dm = qubits + nqubits
        targets = gate.target_qubits
        targets_dm = tuple(q + nqubits for q in targets)
        state = self.cast(state)
        shape = state.shape
        state = self._one_qubit_base(
            state.ravel(), 2 * nqubits, *targets, "apply_y", matrix, qubits_dm
        )
        # force using ``apply_gate`` kernel so that conjugate is properly applied
        state = self._one_qubit_base(
            state, 2 * nqubits, *targets_dm, "apply_gate", self.conj(matrix), qubits
        )
        return self.reshape(state, shape)

    def _as_custom_matrix(self, gate: Gate) -> ArrayLike:
        name = gate.__class__.__name__
        _matrix = getattr(self.custom_matrices, name)

        if name == "FanOut":
            return _matrix(*gate.init_args)

        if isinstance(gate, ParametrizedGate):
            if name == "GeneralizedRBS":  # pragma: no cover
                # this is tested in qibo tests
                theta = gate.init_kwargs["theta"]
                phi = gate.init_kwargs["phi"]
                return _matrix(gate.init_args[0], gate.init_args[1], theta, phi)

            return _matrix(*gate.parameters)

        if isinstance(gate, FusedGate):  # pragma: no cover
            # fusion is tested in qibo tests
            return self.matrix_fused(gate)

        return _matrix(2 ** len(gate.target_qubits)) if callable(_matrix) else _matrix

    def _collapse_statevector(
        self,
        state: ArrayLike,
        qubits: Union[Tuple[int, ...], List[int]],
        shot: int,
        nqubits: int,
        normalize: bool = True,
    ) -> ArrayLike:
        state = self.cast(state, dtype=state.dtype)
        qubits = self.cast(
            [nqubits - q - 1 for q in reversed(qubits)], dtype=self.int32
        )

        if normalize:
            return self.ops.collapse_state_normalized(state, qubits, int(shot), nqubits)

        return self.ops.collapse_state(state, qubits, int(shot), nqubits)

    def _create_qubits_tensor(self, gate: Gate, nqubits: int) -> ArrayLike:
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return self.cast(sorted(qubits), dtype=self.int32)

    def _identity_sparse(
        self, dims: int, dtype: Optional[DTypeLike] = None, **kwargs
    ) -> ArrayLike:  # pragma: no cover
        if dtype is None:
            dtype = self.dtype

        sparsity_format = kwargs.get("format", "csr")

        return eye_sparse(dims, dtype=dtype, format=sparsity_format, **kwargs)

    def _multi_qubit_base(
        self,
        state: ArrayLike,
        nqubits: int,
        targets: Union[List[int], Tuple[int, ...]],
        gate: Gate,
        qubits: Union[List[int], Tuple[int, ...]],
    ) -> ArrayLike:
        if qubits is None:
            qubits = self.cast(
                sorted(nqubits - q - 1 for q in targets), dtype=self.int32
            )
        nstates = 1 << (nqubits - len(qubits))
        targets = self.cast(
            [1 << (nqubits - t - 1) for t in targets[::-1]], dtype=self.int64
        )

        kernel = (
            self.gates.apply_multi_qubit_gate_kernel
            if len(targets) > 5
            else self.multi_qubit_kernels.get(len(targets))
        )

        return kernel(state, gate, qubits, nstates, targets)

    def _one_qubit_base(
        self, state: ArrayLike, nqubits: int, target, kernel, gate, qubits
    ):
        ncontrols = len(qubits) - 1 if qubits is not None else 0
        m = nqubits - target - 1
        nstates = 1 << (nqubits - ncontrols - 1)
        if ncontrols:
            kernel = getattr(self.gates, f"multicontrol_{kernel}_kernel")
            return kernel(state, gate, qubits, nstates, m)
        kernel = getattr(self.gates, f"{kernel}_kernel")
        return kernel(state, gate, nstates, m)

    def _two_qubit_base(
        self, state, nqubits: int, target1, target2, kernel, gate, qubits
    ):
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
            kernel = getattr(self.gates, f"multicontrol_{kernel}_kernel")
            return kernel(state, gate, qubits, nstates, m1, m2, swap_targets)
        kernel = getattr(self.gates, f"{kernel}_kernel")
        return kernel(state, gate, nstates, m1, m2, swap_targets)
