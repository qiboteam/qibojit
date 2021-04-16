from ops.numba import PythonCustomBackend


class CupyCPPBackend(PythonCustomBackend):

    DEFAULT_BLOCK_SIZE = 1024

    def __init__(self):
        import os
        import cupy as cp
        self.cp = cp
        self.op = None

        module_dir = os.path.dirname(os.path.realpath(__file__))
        module_dir = os.path.join(module_dir, "kernels.cu.cc")
        with open(module_dir, "r") as file:
            module = self.cp.RawModule(code=r"{}".format(file.read()))

        kernel_names = [
            "apply_gate_kernel",
            "apply_x_kernel",
            "apply_y_kernel",
            "apply_z_kernel",
            "apply_z_pow_kernel"
        ]
        for name in kernel_names:
            controlled_name = f"multicontrol_{name}"
            setattr(self, name, module.get_function(name))
            setattr(self, controlled_name, module.get_function(controlled_name))

    def cast(self, x, dtype="complex128"):
        return self.cp.asarray(x)

    def create_cache(self, nqubits, targets, controls=[]):
        cache = super().create_cache(nqubits, targets, controls)
        return self.cp.asarray(cache)

    def apply_gate_base(self, state, nqubits, target, kernel, cache=None, gate=None):
        ncontrols = len(cache) - 1 if cache is not None else 0
        m = nqubits - target - 1
        tk = 1 << m
        nstates = 1 << (nqubits - ncontrols - 1)

        block_size = self.DEFAULT_BLOCK_SIZE
        nblocks = (nstates + block_size - 1) // block_size
        if nstates < block_size:
            nblocks = 1
            block_size = nstates

        kernel = f"{kernel}_kernel"
        args = [state, tk, m]
        if gate is not None:
            args.append(gate)
        if ncontrols:
            kernel = f"multicontrol_{kernel}"
            args.append(cache)
            args.append(ncontrols)

        kernel = getattr(self, kernel)
        kernel((nblocks,), (block_size,), tuple(args))
        return state

    def to_numpy(self, x):
        return x.get()
