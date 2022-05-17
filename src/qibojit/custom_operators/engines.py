import numpy as np
from qibo.engines.abstract import Simulator
from qibo.engines.matrices import Matrices


class NumbaMatrices(Matrices):
    # These matrices are used by the custom operators and may 
    # not correspond to the mathematical representation of each gate

    def __init__(self, dtype):
        self.dtype = dtype

    def U1(self, theta):
        return np.array([np.exp(1j * theta)], dtype=self.dtype)

    def CNOT(self):
        return self.X()

    def CZ(self):
        return self.Z()

    # TODO: Implement fSim

    def TOFFOLI(self):
        return self.X()


class NumbaEngine(Simulator):

    def __init__(self, dtype="complex128"):
        from qibojit.custom_operators import gates, ops
        self.dtype = dtype
        self.matrices = NumbaMatrices(dtype)
        self.gates = gates
        self.ops = ops
        self.multi_qubit_kernels = {
            3: self.gates.apply_three_qubit_gate_kernel,
            4: self.gates.apply_four_qubit_gate_kernel,
            5: self.gates.apply_five_qubit_gate_kernel
            }
        self._gate_ops = {
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

    def asmatrix(self, gate):
        return getattr(self.matrices, gate.__class__.__name__)(*gate.parameters)

    def getop(self, gate):
        op = self._gate_ops.get(gate.__class__.__name__)
        if op is None:
            ntargets = len(gate.target_qubits)
            if ntargets == 1:
                return "apply_gate"
            elif ntargets == 2:
                return "apply_two_qubit_gate_kernel"
            else:
                return "apply_multi_qubit_gate"
        else:
            return op

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

    def _create_qubits_tensor(self, gate, nqubits):
        # TODO: Treat density matrices
        qubits = [nqubits - q - 1 for q in gate.control_qubits]
        qubits.extend(nqubits - q - 1 for q in gate.target_qubits)
        return np.array(sorted(qubits), dtype="int32")

    def apply_gate(self, gate, state, nqubits):
        # TODO: Implement density matrices (most likely in another method)
        op = self.getop(gate)
        matrix = self.asmatrix(gate)
        qubits = self._create_qubits_tensor(gate, nqubits)
        targets = gate.target_qubits
        if len(targets) == 1:
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        elif len(targets) == 2:
            return self.one_qubit_base(state, nqubits, *targets, op, matrix, qubits)
        else:
            return self.multi_qubit_base(state, nqubits, targets, op, matrix, qubits)

    def zero_state(self, nqubits, is_matrix=False):
        """Generate |000...0> state as an array."""
        size = 2 ** nqubits
        if is_matrix:
            state = np.empty((size, size), dtype=self.dtype)
            return self.ops.initial_density_matrix(state)
        state = np.empty((size,), dtype=self.dtype)
        return self.ops.initial_state_vector(state)
