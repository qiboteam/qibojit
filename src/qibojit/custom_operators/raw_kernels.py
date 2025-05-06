# This file contains the C++/CUDA implementation of many methods
# defined in gates.py and ops.py.
# The methods in gates.py and ops.py are called by NumbaBackend
# while the functions and kernels here are called by CupyBackend.
from pathlib import Path

with open(Path(__file__).parent / "kernel_helpers.cpp") as file:
    helpers = file.read()


# ---------- KERNELS ----------

apply_gate_kernel = (
    helpers
    + """
extern "C"
__global__ void apply_gate_kernel(<TYPE>* state, long tk, int m, const <TYPE>* gate) {
  apply_gate_kernel<<TYPE>>(state, tk, m, gate);
}
"""
)  # pragma: no cover


apply_x_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void apply_x_kernel(<TYPE>* state, long tk, int m) {
  apply_x_kernel<<TYPE>>(state, tk, m);
}
"""
)  # pragma: no cover


apply_y_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void apply_y_kernel(<TYPE>* state, long tk, int m) {
  apply_y_kernel<<TYPE>>(state, tk, m);
}
"""
)  # pragma: no cover


apply_z_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void apply_z_kernel(<TYPE>* state, long tk, int m) {
  apply_z_kernel<<TYPE>>(state, tk, m);
}
"""
)  # pragma: no cover


apply_z_pow_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_z_pow_kernel()
extern "C"
__global__ void apply_z_pow_kernel(<TYPE>* state, long tk, int m,
                                   const <TYPE>* gate) {
  apply_z_pow_kernel<<TYPE>>(state, tk, m, gate);
}
"""
)  # pragma: no cover


apply_two_qubit_gate_kernel = (
    helpers
    + """
extern "C"
__global__ void apply_two_qubit_gate_kernel(<TYPE>* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const <TYPE>* gate) {
  apply_two_qubit_gate_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2, gate);
}
"""
)  # pragma: no cover


apply_fsim_kernel = (
    helpers
    + """
extern "C"
__global__ void apply_fsim_kernel(<TYPE>* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const <TYPE>* gate) {
  apply_fsim_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2, gate);
}
"""
)  # pragma: no cover


apply_swap_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_swap_kernel()
extern "C"
__global__ void apply_swap_kernel(<TYPE>* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2) {
  apply_swap_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2);
}
"""
)  # pragma: no cover

multicontrol_apply_gate_kernel = (
    helpers
    + """
extern "C"
__global__ void multicontrol_apply_gate_kernel(<TYPE>* state, long tk, int m, const <TYPE>* gate,
                                                      const int* qubits, int ncontrols) {
  multicontrol_apply_gate_kernel<<TYPE>>(state, tk, m, gate, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_x_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void multicontrol_apply_x_kernel(<TYPE>* state, long tk, int m,
                                                      const int* qubits, int ncontrols) {
  multicontrol_apply_x_kernel<<TYPE>>(state, tk, m, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_y_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void multicontrol_apply_y_kernel(<TYPE>* state, long tk, int m,
                                                    const int* qubits, int ncontrols) {
  multicontrol_apply_y_kernel<<TYPE>>(state, tk, m, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_z_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void multicontrol_apply_z_kernel(<TYPE>* state, long tk, int m,
                                                      const int* qubits, int ncontrols) {
  multicontrol_apply_z_kernel<<TYPE>>(state, tk, m, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_z_pow_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_z_pow_kernel()
extern "C"
__global__ void multicontrol_apply_z_pow_kernel(<TYPE>* state, long tk, int m,
                                   const <TYPE>* gate, const int* qubits, int ncontrols) {
  multicontrol_apply_z_pow_kernel<<TYPE>>(state, tk, m, gate, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_two_qubit_gate_kernel = (
    helpers
    + """
extern "C"
__global__ void multicontrol_apply_two_qubit_gate_kernel(<TYPE>* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const <TYPE>* gate,
                                            const int* qubits, int ncontrols) {
  multicontrol_apply_two_qubit_gate_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2, gate, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_fsim_kernel = (
    helpers
    + """
extern "C"
__global__ void multicontrol_apply_fsim_kernel(<TYPE>* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const <TYPE>* gate,
                                  const int* qubits, int ncontrols) {
  multicontrol_apply_fsim_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2, gate, qubits, ncontrols);
}
"""
)  # pragma: no cover


multicontrol_apply_swap_kernel = (
    helpers
    + """
// C++ implementation of gates.py:apply_swap_kernel()
extern "C"
__global__ void multicontrol_apply_swap_kernel(<TYPE>* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const int* qubits, int ncontrols) {
  multicontrol_apply_swap_kernel<<TYPE>>(state, tk1, tk2, m1, m2, uk1, uk2, qubits, ncontrols);
}
"""
)  # pragma: no cover


apply_multi_qubit_gate_kernel = (
    helpers
    + """
extern "C" __global__ void
__launch_bounds__(MAX_BLOCK_SIZE) // to prevent cuda_error_launch_out_of_resources.
                                  // the maximum block size is chosen in backends.py
                                  // and it is replaced here before compilation.
apply_multi_qubit_gate_kernel(<TYPE>* state,
                              const <TYPE>* gate,
                              const int* qubits,
                              const long* targets,
                              int ntargets,
                              int ncontrols) {
  apply_multi_qubit_gate_kernel<<TYPE>>(state, gate, qubits, targets, ntargets, ncontrols)
}
"""
)  # pragma: no cover


collapse_state_kernel = (
    helpers
    + """
extern "C"
__global__ void collapse_state_kernel(<TYPE>* state, const int qubits,
                                      const long result, int ntargets) {
  collapse_state_kernel<<TYPE>>(state, qubits, result, ntargets);
}
"""
)  # pragma: no cover


initial_state_kernel = """
#include <cupy/complex.cuh>

// C++ implementation of ops.py:initial_state_vector()
// In contrast to the Python method, the state is inizialized
// to zero in backends.py:CupyBackend.initial_state, then a
// single thread execute this kernel and set the first element to 1
extern "C" __global__ void initial_state_kernel(<TYPE>* state) {
  <BODY>
}
"""  # pragma: no cover


initial_state_kernel_real = """
// C++ implementation of ops.py:initial_state_vector()
// In contrast to the Python method, the state is inizialized
// to zero in backends.py:CupyBackend.initial_state, then a
// single thread execute this kernel and set the first element to 1
extern "C" __global__ void initial_state_kernel(<TYPE>* state) {
  state[0] = 1;
}
"""  # pragma: no cover
