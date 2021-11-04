// This file contains the C++/CUDA implementation of many methods
// defined in gates.py and ops.py.
// The methods in gates.py and ops.py are called by NumbaBackend
// while the functions and kernels here are called by CupyBackend.

#include <cupy/complex.cuh>


// C++ implementation of gates.py:multicontrol_index()
__device__ long multicontrol_index(const int* qubits, long g, int ncontrols) {
  long i = g;
  for (int iq = 0; iq < ncontrols; iq++) {
      const int n = qubits[iq];
      long k = (long)(1 << n);
      i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  return i;
}


// Helper method for apply_gate_kernel()
template<typename T>
__device__ void _apply_gate(T& state1, T& state2, const T* gate) {
  const T buffer = state1;
  state1 = gate[0] * state1 + gate[1] * state2;
  state2 = gate[2] * buffer + gate[3] * state2;
}


// Helper method for apply_x_kernel()
template<typename T>
__device__ void _apply_x(T& state1, T& state2) {
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}


// Helper method for apply_y_kernel()
template<typename T>
__device__ void _apply_y(T& state1, T& state2) {
  state1 = state1 * T(0, 1);
  state2 = state2 * T(0, -1);
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}


// Helper method for apply_z_kernel()
template<typename T>
__device__ void _apply_z(T& state) {
  state = state * T(-1);
}


// Helper method for apply_z_pow_kernel()
template<typename T>
__device__ void _apply_z_pow(T& state, T gate) {
  state = state * gate;
}


// Helper method for apply_two_qubit_gate_kernel()
template<typename T>
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


// Helper method for apply_fsim_kernel()
template<typename T>
__device__ void _apply_fsim(T& state1, T& state2, T& state3, const T* gate) {
  const T buffer = state1;
  state1 = gate[0] * state1 + gate[1] * state2;
  state2 = gate[2] * buffer + gate[3] * state2;
  state3 = gate[4] * state3;
}


// C++ implementation of gates.py:apply_gate_kernel()
template<typename T>
__global__ void apply_gate_kernel(T* state, long tk, int m, const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_gate<T>(state[i], state[i + tk], gate);
}


// C++ implementation of gates.py:apply_x_kernel()
template<typename T>
__global__ void apply_x_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_x<T>(state[i], state[i + tk]);
}


// C++ implementation of gates.py:apply_y_kernel()
template<typename T>
__global__ void apply_y_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_y<T>(state[i], state[i + tk]);
}


// C++ implementation of gates.py:apply_z_kernel()
template<typename T>
__global__ void apply_z_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z<T>(state[i + tk]);
}


// C++ implementation of gates.py:apply_z_pow_kernel()
template<typename T>
__global__ void apply_z_pow_kernel(T* state, long tk, int m,
                                   const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z_pow<T>(state[i + tk], gate[0]);
}


// C++ implementation of gates.py:apply_two_qubit_gate_kernel()
// The portion of code before the parallel for of the Python
// method is in backends.py:CupyBackend.two_qubit_base()
template<typename T>
__global__ void apply_two_qubit_gate_kernel(T* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_two_qubit_gate<T>(state[i], state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}


// C++ implementation of gates.py:apply_fsim_kernel()
// The portion of code before the parallel for of the Python
// method is in backends.py:CupyBackend.two_qubit_base()
template<typename T>
__global__ void apply_fsim_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_fsim<T>(state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}


// C++ implementation of gates.py:apply_swap_kernel()
template<typename T>
__global__ void apply_swap_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_x<T>(state[i + tk2], state[i + tk1]);
}


// C++ implementation of gates.py:multicontrol_apply_gate_kernel()
template<typename T>
__global__ void multicontrol_apply_gate_kernel(T* state, long tk, int m, const T* gate,
                                               const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_gate<T>(state[i - tk], state[i], gate);
}


// C++ implementation of gates.py:multicontrol_apply_x_kernel()
template<typename T>
__global__ void multicontrol_apply_x_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x<T>(state[i - tk], state[i]);
}


// C++ implementation of gates.py:multicontrol_apply_y_kernel()
template<typename T>
__global__ void multicontrol_apply_y_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_y<T>(state[i - tk], state[i]);
}


// C++ implementation of gates.py:multicontrol_apply_z_kernel()
template<typename T>
__global__ void multicontrol_apply_z_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z<T>(state[i]);
}


// C++ implementation of gates.py:multicontrol_apply_z_pow_kernel()
template<typename T>
__global__ void multicontrol_apply_z_pow_kernel(T* state, long tk, int m,
                                                const T* gate,
                                                const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z_pow<T>(state[i], gate[0]);
}


// C++ implementation of gates.py:multicontrol_apply_two_qubit_gate_kernel()
// The portion of code before the parallel for of the Python method
// is in backends.py:CupyBackend.two_qubit_base()
template<typename T>
__global__ void multicontrol_apply_two_qubit_gate_kernel(T* state,
                                                         long tk1, long tk2,
                                                         int m1, int m2,
                                                         long uk1, long uk2,
                                                         const T* gate,
                                                         const int* qubits,
                                                         int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_two_qubit_gate<T>(state[i - uk1 - uk2], state[i - uk2], state[i - uk1], state[i], gate);
}


// C++ implementation of gates.py:multicontrol_apply_fsim_kernel()
// The portion of code before the parallel for of the Python method
// is in backends.py:CupyBackend.two_qubit_base()
template<typename T>
__global__ void multicontrol_apply_fsim_kernel(T* state,
                                               long tk1, long tk2,
                                               int m1, int m2,
                                               long uk1, long uk2,
                                               const T* gate,
                                               const int* qubits,
                                               int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_fsim<T>(state[i - uk2], state[i - uk1], state[i], gate);
}


// C++ implementation of gates.py:multicontrol_apply_swap_kernel()
template<typename T>
__global__ void multicontrol_apply_swap_kernel(T* state,
                                               long tk1, long tk2,
                                               int m1, int m2,
                                               long uk1, long uk2,
                                               const int* qubits,
                                               int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x<T>(state[i - tk1], state[i - tk2]);
}


// C++ implementation of gates.py:multitarget_index()
__device__ long multitarget_index(const long* targets, long i, int ntargets) {
  long t = 0;
  for (int u = 0; u < ntargets; u++) {
    t += ((long)(i >> u) & 1) * targets[u];
  }
  return t;
}


// C++ implementation of gates.py:apply_multi_qubit_gate_kernel()
template<typename T, int nsubstates>
__global__ void
__launch_bounds__(QIBO_MAX_BLOCK_SIZE) // to prevent CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES.
                                       // The maximum block size is chosen in backends.py
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


// C++ implementation of ops.py:collapse_index()
__device__ long collapse_index(const int* qubits, long g, long h, int ntargets) {
  long i = g;
  for (auto iq = 0; iq < ntargets; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1));
    i += ((long)((int)(h >> iq) % 2) * k);
  }
  return i;
}


// C++ implementation of ops.py:collapse_state() and ops.py:collapse_state_normalized()
// Only the parallel for is implemented here. The other portions of code are
// implemented in backends.py:CupyBackend.collapse_state()
template <typename T>
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


// C++ implementation of ops.py:initial_state_vector()
// In contrast to the Python method, the state is inizialized
// to zero in backends.py:CupyBackend.initial_state, then a
// single thread execute this kernel and set the first element to 1
template <typename T>
__global__ void initial_state_kernel(T* state) {
  state[0] = T(1, 0);
}
