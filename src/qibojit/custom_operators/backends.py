from abc import ABC, abstractmethod


class AbstractBackend(ABC):

    def __init__(self):
        self.name = "abstract"
        self.gates = None
        self.ops = None

    def cast(self, x, dtype=None):
        return x

    def to_numpy(self, x):
        return x

    @abstractmethod
    def one_qubit_base(self, state, nqubits, target, kernel, qubits=None, gate=None):
        raise NotImplementedError

    @abstractmethod
    def two_qubit_base(self, state, nqubits, target1, target2, kernel, qubits=None, gate=None):
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, nqubits, dtype, is_matrix=False):
        raise NotImplementedError

    @abstractmethod
    def collapse_state(self, state, qubits, result, nqubits, normalize):
        raise NotImplementedError

    @abstractmethod
    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        raise NotImplementedError


class NumbaBackend(AbstractBackend):

    def __init__(self):
        import numpy as np
        from qibojit.custom_operators import gates, ops
        self.name = "numba"
        self.gates = gates
        self.ops = ops
        self.np = np

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

    def initial_state(self, nqubits, dtype, is_matrix=False):
        if isinstance(dtype, str):
            dtype = getattr(self.np, dtype)
        if is_matrix:
            return self.ops.initial_density_matrix(nqubits, dtype)
        return self.ops.initial_state_vector(nqubits, dtype)

    def collapse_state(self, state, qubits, result, nqubits, normalize):
        return self.ops.collapse_state(state, qubits, result, nqubits, normalize)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        return self.ops.measure_frequencies(frequencies, probs, nshots, nqubits, seed)


class CupyBackend(AbstractBackend):

    DEFAULT_BLOCK_SIZE = 1024
    KERNELS = ("apply_gate", "apply_x", "apply_y", "apply_z", "apply_z_pow",
               "apply_two_qubit_gate", "apply_fsim", "apply_swap")

    def __init__(self):
        import os
        import cupy as cp
        self.name = "cupy"
        self.cp = cp
        base_dir = os.path.dirname(os.path.realpath(__file__))

        # load gate kernels
        kernels = []
        for kernel in self.KERNELS:
            kernels.append(f"{kernel}_kernel<complex<double>>")
            kernels.append(f"{kernel}_kernel<complex<float>>")
            kernels.append(f"multicontrol_{kernel}_kernel<complex<double>>")
            kernels.append(f"multicontrol_{kernel}_kernel<complex<float>>")
        kernels = tuple(kernels)
        gates_dir = os.path.join(base_dir, "gates.cu.cc")
        with open(gates_dir, "r") as file:
            code = r"{}".format(file.read())
            self.gates = cp.RawModule(code=code, options=("--std=c++11",),
                                      name_expressions=kernels)

        # load `collapse_state` kernels
        kernels = tuple(
            "collapse_state_kernel<complex<double>>",
            "collapse_state_kernel<complex<float>>",
            "collapsed_norm_kernel<complex<double>,double>",
            "collapsed_norm_kernel<complex<float>,float>",
            "vector_reduction_kernel<double>",
            "vector_reduction_kernel<float>",
            "normalize_collapsed_state_kernel<complex<double>,double>",
            "normalize_collapsed_state_kernel<complex<float>,float>"
        )
        ops_dir = os.path.join(base_dir, "ops.cu.cc")
        with open(ops_dir, "r") as file:
            code = r"{}".format(file.read())
            self.ops = cp.RawModule(code=code, options=("--std=c++11",),
                                    name_expressions=kernels)

    def calculate_blocks(self, nstates):
        block_size = self.DEFAULT_BLOCK_SIZE
        nblocks = (nstates + block_size - 1) // block_size
        if nstates < block_size:
            nblocks = 1
            block_size = nstates
        return nblocks, block_size

    def cast(self, x, dtype=None):
        if isinstance(x, self.cp.ndarray):
            return x
        return self.cp.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return x.get()

    def get_kernel_type(self, state):
        if state.dtype == self.cp.complex128:
            return "complex<double>"
        elif state.dtype == self.cp.complex64:
            return "complex<float>"
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
            args = (state, tk, m, self.cast(gate))
            assert state.dtype == args[-1].dtype

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel<{ktype}>")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 1)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel<{ktype}>")

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
            args = (state, tk1, tk2, m1, m2, uk1, uk2, self.cast(gate))
            assert state.dtype == args[-1].dtype

        ktype = self.get_kernel_type(state)
        if ncontrols:
            kernel = self.gates.get_function(f"multicontrol_{kernel}_kernel<{ktype}>")
            args += (self.cast(qubits, dtype=self.cp.int32), ncontrols + 2)
        else:
            kernel = self.gates.get_function(f"{kernel}_kernel<{ktype}>")

        nblocks, block_size = self.calculate_blocks(nstates)
        kernel((nblocks,), (block_size,), args)
        self.cp.cuda.stream.get_current_stream().synchronize()
        return state

    def initial_state(self, nqubits, dtype, is_matrix=False):
        n = 1 << nqubits
        if is_matrix:
            state = self.cp.zeros((n, n), dtype=dtype)
            state[0, 0] = 1
        else:
            state = self.cp.zeros(n, dtype=dtype)
            state[0] = 1
        return state

    def collapse_state(self, state, qubits, result, nqubits, normalize=True):
        ntargets = len(qubits)
        nstates = 1 << (nqubits - ntargets)
        int blockSize = DEFAULT_BLOCK_SIZE;
        nblocks, block_size = self.calculate_blocks(nstates)

        ktype = self.get_kernel_type(state)
        args = [state, qubits, result, ntargets]
        # TODO: Implemenet `collapse_state_kernel`
        kernel = self.ops.get_function(f"collapse_state_kernel<{ktype}>")
        kernel((nblocks,), (block_size,), args)

        if normalize:
            # TODO: Check if it is faster to do this with `cp` primitives
            # instead of custom kernels
            rtype = ktype.split("<")[1][:-1]
            # allocate support arrays on GPU
            if rtype == "double":
                # norms = 0
                # not sure if `nblocks` is the proper shape here
                block_norms = cp.zeros(nblocks, dtype="float64")
            else:
                # norms = 0
                block_norms = cp.zeros(nblocks, dtype="float32")

            args.append(nstates)
            args.append(block_norms)
            kernel = self.ops.get_function(f"collapsed_norm_kernel<{ktype},{rtype}>")
            kernel((1,), (block_size,), args)
            # TODO: check if it is faster to do this calculation using custom kernel
            #kernel = self.gates.get_function(f"vector_reduction_kernel<{rtype}>")
            #kernel((1,), (block_size,), [block_norms, norms])
            norms = cp.sum(block_norms)
            args.pop()
            args.append(norms)
            kernel = self.ops.get_function(f"normalize_collapsed_state_kernel<{ktype},{rtype}>")
            kernel((nblocks,), (block_size,), args)

    def measure_frequencies(self, frequencies, probs, nshots, nqubits, seed=1234):
        raise NotImplementedError("`measure_frequencies` method is not "
                                  "implemented for GPU.")
