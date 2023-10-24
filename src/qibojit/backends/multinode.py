"""Support execution on multiple nodes.

The implementation is heavily inspired by the
:cls:`qibojit.backends.gpu.MultiGpuOps` backend for multiple GPUs.

.. todo::

    The only thing specific to GPU of the `MultiGpuOps` backend is the way the devices are
    enumerated in the `apply_gates` method.
    Thus, it should be possible to reimplement it just using the `MultiNode` and a
    suitable scheduler (i.e. a helper function will generate the scheduler to emulate
    the `MultiGpuOps` backend).
"""

import numpy as np


class MultiNode:  # pragma: no cover
    """Multinode execution backend.

    It relies on a `local_backend` to execute the operations on the various nodes, and a
    `global_backend` for the operations executed by the master.
    If the `global_backend` is not specified it defaults to the local one.

    .. todo::

        - turn `local_backend` into a map of `node: backend`, to allow a different
          backend for each node (this would allow for GPUs to be treated as regular
          nodes, e.g.)
        - write constructor to allow for a node list and a single local backend
    """

    def __init__(self, local_backend, circuit, global_backend=None):
        self.backend = local_backend
        self.circuit = circuit
        self.global_ops = (
            global_backend if global_backend is not None else local_backend
        ).ops

    def transpose_state(self, pieces, state, nqubits, order):
        """Transpose state.

        This operation is directly executed on the master.
        """
        original_shape = state.shape
        state = state.ravel()
        state = self.global_ops.transpose_state(tuple(pieces), state, nqubits, order)
        return np.reshape(state, original_shape)

    def to_pieces(self, state):
        """Split state to pieces."""
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
        """Reconstruct tensor from pieces."""
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
        """Execute special gates on CPU.

        Note
        ----
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
        """Execute qubits swap on pieces."""
        m = (
            self.circuit.nglobal
            - self.circuit.queues.qubits.reduced_global.get(global_qubit)
            - 1
        )
        t = 1 << m
        for g in range(self.circuit.ndevices // 2):
            i = ((g >> m) << (m + 1)) + (g & (t - 1))
            local_eff = self.circuit.queues.qubits.reduced_local.get(local_qubit)
            self.global_ops.swap_pieces(
                pieces[i], pieces[i + t], local_eff, self.circuit.nlocal
            )
        return pieces

    def revert_swaps(self, pieces, swap_pairs):
        """Revert qubit swaps."""
        for q1, q2 in swap_pairs:
            if q1 not in self.circuit.queues.qubits.set:
                q1, q2 = q2, q1
            pieces = self.swap(pieces, q1, q2)
        return pieces
