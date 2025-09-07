import sys

from collections import Counter

from qibo.backends import Backend, NumpyBackend
from qibo.gates.abstract import ParametrizedGate
from qibo.gates.special import FusedGate

from qibojit.backends.matrices import CustomMatrices

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


class NumbaBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        import psutil  # pylint: disable=import-outside-toplevel

        import numpy as np

        from numba import (
            __version__ as numba_version,
        )  # pylint: disable=import-outside-toplevel

        from qibojit import (
            __version__ as qibojit_version,
        )  # pylint: disable=import-outside-toplevel
        from qibojit.custom_operators import (
            gates,
            ops,
        )  # pylint: disable=import-outside-toplevel

        self.custom_matrices = CustomMatrices(self.dtype)
        self.device = "/CPU:0"
        self.engine = np
        self.gates = gates
        self.name = "qibojit"
        self.multi_qubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel,
        }
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.ops = ops
        self.platform = "numba"
        self.tensor_types = (np.ndarray,)
        self.versions.update(
            {
                "qibojit": qibojit_version,
                "numba": numba_version,
            }
        )

        self.measure_frequencies_op = self.ops.measure_frequencies

        if sys.platform == "darwin":  # pragma: no cover
            self.set_threads(psutil.cpu_count(logical=False))
        else:
            self.set_threads(len(psutil.Process().cpu_affinity()))

    def set_dtype(self, dtype):
        if dtype != self.dtype:
            super().set_dtype(dtype)
            if self.custom_matrices:
                self.custom_matrices = CustomMatrices(self.dtype)

    def set_threads(self, nthreads):
        import numba  # pylint: disable=import-outside-toplevel

        numba.set_num_threads(nthreads)
        self.nthreads = nthreads

    def zero_state(self, nqubits, density_matrix: bool = False, dtype=None):
        if dtype is None:
            dtype = self.dtype

        size = 2**nqubits
        shape = (size, size) if density_matrix else (size,)
        state = self.empty(shape, dtype=dtype)

        func = (
            self.ops.initial_density_matrix
            if density_matrix
            else self.ops.initial_state_vector
        )

        return func(state)

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
            qubits = self.cast(sorted(nqubits - q - 1 for q in targets), dtype=self.int32)
        nstates = 1 << (nqubits - len(qubits))
        targets = self.cast(
            [1 << (nqubits - t - 1) for t in targets[::-1]], dtype=self.int64
        )
        if len(targets) > 5:
            kernel = self.gates.apply_multi_qubit_gate_kernel
        else:
            kernel = self.multi_qubit_kernels.get(len(targets))
        return kernel(state, gate, qubits, nstates, targets)

    def _create_qubits_tensor(self, gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return self.cast(sorted(qubits), dtype=self.int32)

    def _as_custom_matrix(self, gate):
        name = gate.__class__.__name__
        _matrix = getattr(self.custom_matrices, name)

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
            state = self.one_qubit_base(
                state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm
            )
            state = self.one_qubit_base(
                state, 2 * nqubits, *targets_dm, op, self.conj(matrix), qubits
            )
        elif len(targets) == 2:
            op = GATE_OPS.get(name, "apply_two_qubit_gate")
            state = self.two_qubit_base(
                state.ravel(), 2 * nqubits, *targets, op, matrix, qubits_dm
            )
            state = self.two_qubit_base(
                state, 2 * nqubits, *targets_dm, op, self.conj(matrix), qubits
            )
        else:
            state = self.multi_qubit_base(
                state.ravel(), 2 * nqubits, targets, matrix, qubits_dm
            )
            state = self.multi_qubit_base(
                state, 2 * nqubits, targets_dm, self.conj(matrix), qubits
            )
        return self.reshape(state, shape)

    def _apply_ygate_density_matrix(self, gate, state, nqubits):
        matrix = self._as_custom_matrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        qubits_dm = qubits + nqubits
        targets = gate.target_qubits
        targets_dm = tuple(q + nqubits for q in targets)
        state = self.cast(state)
        shape = state.shape
        state = self.one_qubit_base(
            state.ravel(), 2 * nqubits, *targets, "apply_y", matrix, qubits_dm
        )
        # force using ``apply_gate`` kernel so that conjugate is properly applied
        state = self.one_qubit_base(
            state, 2 * nqubits, *targets_dm, "apply_gate", self.conj(matrix), qubits
        )
        return self.reshape(state, shape)

    def apply_channel_density_matrix(self, channel, state, nqubits):
        state = self.cast(state)
        if not channel._all_unitary_operators:
            state_copy = self.cast(state, copy=True)
        new_state = (1 - channel.coefficient_sum) * state
        for coeff, gate in zip(channel.coefficients, channel.gates):
            state = self.apply_gate_density_matrix(gate, state, nqubits)
            new_state += coeff * state
            # reset the state
            if not channel._all_unitary_operators:
                state = self.cast(state_copy, copy=True)
            else:
                state = self.apply_gate_density_matrix(
                    gate, state, nqubits, inverse=True
                )
        return new_state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        qubits = self.cast([nqubits - q - 1 for q in reversed(qubits)], dtype="int32")
        if normalize:
            return self.ops.collapse_state_normalized(state, qubits, int(shot), nqubits)
        else:
            return self.ops.collapse_state(state, qubits, int(shot), nqubits)

    def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        dm_qubits = [q + nqubits for q in qubits]
        state = self.collapse_state(state.ravel(), dm_qubits, shot, 2 * nqubits, False)
        state = self.collapse_state(state, qubits, shot, 2 * nqubits, False)
        state = self.np.reshape(state, shape)
        if normalize:
            state = state / self.np.trace(state)
        return state

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_METROPOLIS_THRESHOLD

        if nshots < SHOT_METROPOLIS_THRESHOLD:
            return super().sample_frequencies(probabilities, nshots)

        seed = self.random_integers(0, int(1e8), dtype=self.int64)
        nqubits = int(self.log2(tuple(probabilities.shape)[0]))
        frequencies = self.zeros(2**nqubits, dtype=self.int64)
        # always fall back to numba CPU backend because for ops not implemented on GPU
        frequencies = self.measure_frequencies_op(
            frequencies, probabilities, nshots, nqubits, seed, self.nthreads
        )
        return Counter({i: f for i, f in enumerate(frequencies) if f > 0})
