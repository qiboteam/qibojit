# This file contains the C++/CUDA implementation of many methods
# defined in gates.py and ops.py.
# The methods in gates.py and ops.py are called by NumbaBackend
# while the functions and kernels here are called by CupyBackend.

# ---------- HELPER METHODS ----------

multicontrol_index = """
// C++ implementation of gates.py:multicontrol_index()
__device__ long multicontrol_index(const int* qubits, long g, int ncontrols) {
  long i = g;
  for (int iq = 0; iq < ncontrols; iq++) {
      const int n = qubits[iq];
      long k = ((long)1 << n);
      i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  return i;
}
"""  # pragma: no cover


_apply_gate = """
// Helper method for apply_gate_kernel()
__device__ void _apply_gate(T& state1, T& state2, const T* gate) {
  const T buffer = state1;
  state1 = gate[0] * state1 + gate[1] * state2;
  state2 = gate[2] * buffer + gate[3] * state2;
}
"""  # pragma: no cover


_apply_x = """
// Helper method for apply_x_kernel()
__device__ void _apply_x(T& state1, T& state2) {
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}
"""  # pragma: no cover


_apply_y = """
// Helper method for apply_y_kernel()
__device__ void _apply_y(T& state1, T& state2) {
  state1 = state1 * T(0, 1);
  state2 = state2 * T(0, -1);
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}
"""  # pragma: no cover


_apply_z = """
// Helper method for apply_z_kernel()
__device__ void _apply_z(T& state) {
  state = state * T(-1);
}
"""  # pragma: no cover


_apply_z_pow = """
// Helper method for apply_z_pow_kernel()
__device__ void _apply_z_pow(T& state, T gate) {
  state = state * gate;
}
"""  # pragma: no cover


_apply_two_qubit_gate = """
// Helper method for apply_two_qubit_gate_kernel()
__device__ void _apply_two_qubit_gate(T& state0, T& state1, T& state2, T& state3,
                                      const T* gate) {
  const T buffer0 = state0;
  const T buffer1 = state1;
  const T buffer2 = state2;
  state0 = gate[0]  * state0  + gate[1]  * state1
         + gate[2]  * state2  + gate[3]  * state3;
  state1 = gate[4]  * buffer0 + gate[5]  * state1
         + gate[6]  * state2  + gate[7]  * state3;
  state2 = gate[8]  * buffer0 + gate[9]  * buffer1
         + gate[10] * state2  + gate[11] * state3;
  state3 = gate[12] * buffer0 + gate[13] * buffer1
         + gate[14] * buffer2 + gate[15] * state3;
}
"""  # pragma: no cover


_apply_fsim = """
// Helper method for apply_fsim_kernel()
__device__ void _apply_fsim(T& state1, T& state2, T& state3, const T* gate) {
  const T buffer = state1;
  state1 = gate[0] * state1 + gate[1] * state2;
  state2 = gate[2] * buffer + gate[3] * state2;
  state3 = gate[4] * state3;
}
"""  # pragma: no cover


multitarget_index = """
// C++ implementation of gates.py:multitarget_index()
__device__ long multitarget_index(const long* targets, long i, int ntargets) {
  long t = 0;
  for (int u = 0; u < ntargets; u++) {
    t += ((long)(i >> u) & 1) * targets[u];
  }
  return t;
}
"""  # pragma: no cover


collapse_index = """
// C++ implementation of ops.py:collapse_index()
__device__ long collapse_index(const int* qubits, long g, long h, int ntargets) {
  long i = g;
  for (auto iq = 0; iq < ntargets; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1));
    i += ((long)((long)(h >> iq) % 2) * k);
  }
  return i;
}
"""  # pragma: no cover

# ---------- KERNELS ----------

apply_gate_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_gate}"""
    + """
// C++ implementation of gates.py:apply_gate_kernel()
extern "C"
__global__ void apply_gate_kernel(T* state, long tk, int m, const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_gate(state[i], state[i + tk], gate);
}
"""
)  # pragma: no cover


apply_x_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_x}"""
    + """
// C++ implementation of gates.py:apply_x_kernel()
extern "C"
__global__ void apply_x_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_x(state[i], state[i + tk]);
}
"""
)  # pragma: no cover


apply_y_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_y}"""
    + """
// C++ implementation of gates.py:apply_y_kernel()
extern "C"
__global__ void apply_y_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_y(state[i], state[i + tk]);
}
"""
)  # pragma: no cover


apply_z_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_z}"""
    + """
// C++ implementation of gates.py:apply_z_kernel()
extern "C"
__global__ void apply_z_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z(state[i + tk]);
}
"""
)  # pragma: no cover


apply_z_pow_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_z_pow}"""
    + """
// C++ implementation of gates.py:apply_z_pow_kernel()
extern "C"
__global__ void apply_z_pow_kernel(T* state, long tk, int m,
                                   const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z_pow(state[i + tk], gate[0]);
}
"""
)  # pragma: no cover


apply_two_qubit_gate_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_two_qubit_gate}"""
    + """
// C++ implementation of gates.py:apply_two_qubit_gate_kernel()
// the portion of code before the parallel for of the Python
// method is in backends.py:CupyBackend.two_qubit_base()
extern "C"
__global__ void apply_two_qubit_gate_kernel(T* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_two_qubit_gate(state[i], state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}
"""
)  # pragma: no cover


apply_fsim_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_fsim}"""
    + """
// C++ implementation of gates.py:apply_fsim_kernel()
// the portion of code before the parallel for of the Python
// method is in backends.py:CupyBackend.two_qubit_base()
extern "C"
__global__ void apply_fsim_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_fsim(state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}
"""
)  # pragma: no cover


apply_swap_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_x}"""
    + """
// C++ implementation of gates.py:apply_swap_kernel()
extern "C"
__global__ void apply_swap_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_x(state[i + tk2], state[i + tk1]);
}
"""
)  # pragma: no cover


multicontrol_apply_gate_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_gate}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_gate_kernel()
extern "C"
__global__ void multicontrol_apply_gate_kernel(T* state, long tk, int m, const T* gate,
                                               const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_gate(state[i - tk], state[i], gate);
}
"""
)  # pragma: no cover


multicontrol_apply_x_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_x}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_x_kernel()
extern "C"
__global__ void multicontrol_apply_x_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x(state[i - tk], state[i]);
}
"""
)  # pragma: no cover


multicontrol_apply_y_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_y}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_y_kernel()
extern "C"
__global__ void multicontrol_apply_y_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_y(state[i - tk], state[i]);
}
"""
)  # pragma: no cover


multicontrol_apply_z_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_z}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_z_kernel()
extern "C"
__global__ void multicontrol_apply_z_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z(state[i]);
}
"""
)  # pragma: no cover


multicontrol_apply_z_pow_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_z_pow}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_z_pow_kernel()
extern "C"
__global__ void multicontrol_apply_z_pow_kernel(T* state, long tk, int m,
                                                const T* gate,
                                                const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z_pow(state[i], gate[0]);
}
"""
)  # pragma: no cover


multicontrol_apply_two_qubit_gate_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_two_qubit_gate}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_two_qubit_gate_kernel()
// the portion of code before the parallel for of the Python method
// is in backends.py:CupyBackend.two_qubit_base()
extern "C"
__global__ void multicontrol_apply_two_qubit_gate_kernel(T* state,
                                                         long tk1, long tk2,
                                                         int m1, int m2,
                                                         long uk1, long uk2,
                                                         const T* gate,
                                                         const int* qubits,
                                                         int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_two_qubit_gate(state[i - uk1 - uk2], state[i - uk2], state[i - uk1], state[i], gate);
}
"""
)  # pragma: no cover


multicontrol_apply_fsim_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_fsim}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_fsim_kernel()
// the portion of code before the parallel for of the Python method
// is in backends.py:CupyBackend.two_qubit_base()
extern "C"
__global__ void multicontrol_apply_fsim_kernel(T* state,
                                               long tk1, long tk2,
                                               int m1, int m2,
                                               long uk1, long uk2,
                                               const T* gate,
                                               const int* qubits,
                                               int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_fsim(state[i - uk2], state[i - uk1], state[i], gate);
}
"""
)  # pragma: no cover


multicontrol_apply_swap_kernel = (
    f"""
#include <cupy/complex.cuh>
{_apply_x}
{multicontrol_index}"""
    + """
// C++ implementation of gates.py:multicontrol_apply_swap_kernel()
extern "C"
__global__ void multicontrol_apply_swap_kernel(T* state,
                                               long tk1, long tk2,
                                               int m1, int m2,
                                               long uk1, long uk2,
                                               const int* qubits,
                                               int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x(state[i - tk1], state[i - tk2]);
}
"""
)  # pragma: no cover


apply_multi_qubit_gate_kernel = (
    f"""
#include <cupy/complex.cuh>
{multicontrol_index}
{multitarget_index}"""
    + """
// C++ implementation of gates.py:apply_multi_qubit_gate_kernel()
extern "C" __global__ void
__launch_bounds__(MAX_BLOCK_SIZE) // to prevent cuda_error_launch_out_of_resources.
                                  // the maximum block size is chosen in backends.py
                                  // and it is replaced here before compilation.
apply_multi_qubit_gate_kernel(T* state,
                              const T* gate,
                              const int* qubits,
                              const long* targets,
                              int ntargets,
                              int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long ig = multicontrol_index(qubits, g, ncontrols);
  T buffer[nsubstates];
  for (auto i = 0; i < nsubstates; i++) {
    const long t = ig - multitarget_index(targets, nsubstates - i - 1, ntargets);
    buffer[i] = state[t];
  }
  for (auto i = 0; i < nsubstates; i++) {
    const long t = ig - multitarget_index(targets, nsubstates - i - 1, ntargets);
    T new_state_elem = T(0., 0.); // use local variable because it is faster than global ones
    for (auto j = 0; j < nsubstates; j++) {
      new_state_elem += gate[nsubstates * i + j] * buffer[j];
    }
    state[t] = new_state_elem;
  }
}
"""
)  # pragma: no cover


collapse_state_kernel = (
    f"""
#include <cupy/complex.cuh>
{collapse_index}"""
    + """
// C++ implementation of ops.py:collapse_state() and ops.py:collapse_state_normalized()
// Only the parallel for is implemented here. the other portions of code are
// implemented in backends.py:CupyBackend.collapse_state()
extern "C"
__global__ void collapse_state_kernel(T* state, const int* qubits,
                                      const long result, int ntargets) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const long nsubstates = (long)1 << ntargets;
  for (auto h = 0; h < result; h++) {
    state[collapse_index(qubits, g, h, ntargets)] = T(0, 0);
  }
  for (auto h = result + 1; h < nsubstates; h++) {
    state[collapse_index(qubits, g, h, ntargets)] = T(0, 0);
  }
}
"""
)  # pragma: no cover


initial_state_kernel = """
#include <cupy/complex.cuh>

// C++ implementation of ops.py:initial_state_vector()
// In contrast to the Python method, the state is inizialized
// to zero in backends.py:CupyBackend.initial_state, then a
// single thread execute this kernel and set the first element to 1
extern "C" __global__ void initial_state_kernel(T* state) {
  state[0] = T(1, 0);
}
"""  # pragma: no cover
