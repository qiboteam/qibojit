#include <cupy/complex.cuh>


__device__ complex<double> cmult(complex<double> a, complex<double> b) {
  return complex<double>(a.real() * b.real() - a.imag() * b.imag(),
                         a.real() * b.imag() + a.imag() * b.real());
}

__device__ complex<float> cmult(complex<float> a, complex<float> b) {
  return complex<float>(a.real() * b.real() - a.imag() * b.imag(),
                        a.real() * b.imag() + a.imag() * b.real());
}

__device__ complex<double> cadd(complex<double> a, complex<double> b) {
  return complex<double>(a.real() + b.real(), a.imag() + b.imag());
}

__device__ complex<float> cadd(complex<float> a, complex<float> b) {
  return complex<float>(a.real() + b.real(), a.imag() + b.imag());
}

__device__ long multicontrol_index(const int* qubits, long g, int ncontrols) {
  long i = g;
  for (int iq = 0; iq < ncontrols; iq++) {
      const int n = qubits[iq];
      long k = (long)(1 << n);
      i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1)) + k;
  }
  return i;
}


template<typename T>
__device__ void _apply_gate(T& state1, T& state2, const T* gate) {
  const T buffer = state1;
  state1 = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state2 = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

template<typename T>
__device__ void _apply_x(T& state1, T& state2) {
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}

template<typename T>
__device__ void _apply_y(T& state1, T& state2) {
  state1 = cmult(state1, T(0, 1));
  state2 = cmult(state2, T(0, -1));
  const T buffer = state1;
  state1 = state2;
  state2 = buffer;
}

template<typename T>
__device__ void _apply_z(T& state) {
  state = cmult(state, T(-1));
}

template<typename T>
__device__ void _apply_z_pow(T& state, T gate) {
  state = cmult(state, gate);
}

template<typename T>
__device__ void _apply_two_qubit_gate(T& state0, T& state1, T& state2, T& state3,
                                      const T* gate) {
  const T buffer0 = state0;
  const T buffer1 = state1;
  const T buffer2 = state2;
  state0 = cadd(cadd(cmult(gate[0], state0), cmult(gate[1], state1)),
                cadd(cmult(gate[2], state2), cmult(gate[3], state3)));
  state1 = cadd(cadd(cmult(gate[4], buffer0), cmult(gate[5], state1)),
                cadd(cmult(gate[6], state2), cmult(gate[7], state3)));
  state2 = cadd(cadd(cmult(gate[8], buffer0), cmult(gate[9], buffer1)),
                cadd(cmult(gate[10], state2), cmult(gate[11], state3)));
  state3 = cadd(cadd(cmult(gate[12], buffer0), cmult(gate[13], buffer1)),
                cadd(cmult(gate[14], buffer2), cmult(gate[15], state3)));
}

template<typename T>
__device__ void _apply_fsim(T& state1, T& state2, T& state3, const T* gate) {
  const T buffer = state1;
  state1 = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state2 = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
  state3 = cmult(gate[4], state3);
}

template<typename T>
__global__ void apply_gate_kernel(T* state, long tk, int m, const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_gate<T>(state[i], state[i + tk], gate);
}

template<typename T>
__global__ void apply_x_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_x<T>(state[i], state[i + tk]);
}

template<typename T>
__global__ void apply_y_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_y<T>(state[i], state[i + tk]);
}

template<typename T>
__global__ void apply_z_kernel(T* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z<T>(state[i + tk]);
}

template<typename T>
__global__ void apply_z_pow_kernel(T* state, long tk, int m,
                                   const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z_pow<T>(state[i + tk], gate[0]);
}

template<typename T>
__global__ void apply_two_qubit_gate_kernel(T* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_two_qubit_gate<T>(state[i], state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}

template<typename T>
__global__ void apply_fsim_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2,
                                  const T* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_fsim<T>(state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}

template<typename T>
__global__ void apply_swap_kernel(T* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_x<T>(state[i + tk2], state[i + tk1]);
}

template<typename T>
__global__ void multicontrol_apply_gate_kernel(T* state, long tk, int m, const T* gate,
                                               const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_gate<T>(state[i - tk], state[i], gate);
}

template<typename T>
__global__ void multicontrol_apply_x_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x<T>(state[i - tk], state[i]);
}

template<typename T>
__global__ void multicontrol_apply_y_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_y<T>(state[i - tk], state[i]);
}

template<typename T>
__global__ void multicontrol_apply_z_kernel(T* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z<T>(state[i]);
}

template<typename T>
__global__ void multicontrol_apply_z_pow_kernel(T* state, long tk, int m,
                                                const T* gate,
                                                const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z_pow<T>(state[i], gate[0]);
}

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


__device__ long multitarget_index(const long* targets, long i, int ntargets) {
  long t = 0;
  for (int u = 0; u < ntargets; u++) {
    t += ((long)(i >> u) & 1) * targets[u];
  }
  return t;
}


template<typename T>
__global__ void apply_three_qubit_gate_kernel(T* state,
                                              const T* gate,
                                              const int* qubits,
                                              const long* targets,
                                              int ntargets,
                                              int ncontrols) {
  // Compute the thread ID
  const long g = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the base index with control qubits in |1>
  const long ig = multicontrol_index(qubits, g, ncontrols);

  // Initialize the buffer for the in-place edit of the state vector
  const T buffer0 = state[ig - targets[0] - targets[1] - targets[2]];
  const T buffer1 = state[ig - targets[1] - targets[2]];
  const T buffer2 = state[ig - targets[0] - targets[2]];
  const T buffer3 = state[ig - targets[2]];
  const T buffer4 = state[ig - targets[0] - targets[1]];
  const T buffer5 = state[ig - targets[1]];
  const T buffer6 = state[ig - targets[0]];
  const T buffer7 = state[ig];

  // Apply the gate to the state vector
  // Gate here is a 8x8 matrix
  // Iterate on the 8 output elements indexed by t
  for(auto i = 0; i < 8; i++) {
    // Get the index where the state vector must be updated
    const long t = ig - multitarget_index(targets, 7 - i, ntargets);

    // Update the state vector at index t
    state[t] = gate[i*8]   * buffer0 + gate[i*8+1] * buffer1 + gate[i*8+2] * buffer2
             + gate[i*8+3] * buffer3 + gate[i*8+4] * buffer4 + gate[i*8+5] * buffer5
             + gate[i*8+6] * buffer6 + gate[i*8+7] * buffer7;
  }
}


template<typename T>
__global__ void apply_four_qubit_gate_kernel(T* state,
                                             const T* gate,
                                             const int* qubits,
                                             const long* targets,
                                             int ntargets,
                                             int ncontrols) {
  // Compute the thread ID
  const long g = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the base index with control qubits in |1>
  const long ig = multicontrol_index(qubits, g, ncontrols);

  // Initialize the buffer for the in-place edit of the state vector
  const T buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]];
  const T buffer1 = state[ig - targets[1] - targets[2] - targets[3]];
  const T buffer2 = state[ig - targets[0] - targets[2] - targets[3]];
  const T buffer3 = state[ig - targets[2] - targets[3]];
  const T buffer4 = state[ig - targets[0] - targets[1] - targets[3]];
  const T buffer5 = state[ig - targets[1] - targets[3]];
  const T buffer6 = state[ig - targets[0] - targets[3]];
  const T buffer7 = state[ig - targets[3]];
  const T buffer8 = state[ig - targets[0] - targets[1] - targets[2]];
  const T buffer9 = state[ig - targets[1] - targets[2]];
  const T buffer10 = state[ig - targets[0] - targets[2]];
  const T buffer11 = state[ig - targets[2]];
  const T buffer12 = state[ig - targets[0] - targets[1]];
  const T buffer13 = state[ig - targets[1]];
  const T buffer14 = state[ig - targets[0]];
  const T buffer15 = state[ig];

  // Apply the gate to the state vector
  // Gate here is a 16x16 matrix
  // Iterate on the 16 output elements indexed by t
  for(auto i = 0; i < 16; i++) {
    // Get the index where the state vector must be updated
    const long t = ig - multitarget_index(targets, 15 - i, ntargets);

    // Update the state vector at index t
    state[t] = gate[i*16]    * buffer0  + gate[i*16+1]  * buffer1  + gate[i*16+2]  * buffer2
             + gate[i*16+3]  * buffer3  + gate[i*16+4]  * buffer4  + gate[i*16+5]  * buffer5
             + gate[i*16+6]  * buffer6  + gate[i*16+7]  * buffer7  + gate[i*16+8]  * buffer8
             + gate[i*16+9]  * buffer9  + gate[i*16+10] * buffer10 + gate[i*16+11] * buffer11
             + gate[i*16+12] * buffer12 + gate[i*16+13] * buffer13 + gate[i*16+14] * buffer14
             + gate[i*16+15] * buffer15;
  }
}


template<typename T>
__global__ void apply_five_qubit_gate_kernel(T* state,
                                             const T* gate,
                                             const int* qubits,
                                             const long* targets,
                                             int ntargets,
                                             int ncontrols) {
  // Compute the thread ID
  const long g = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the base index with control qubits in |1>
  const long ig = multicontrol_index(qubits, g, ncontrols);

  // Initialize the buffer for the in-place edit of the state vector
  const T buffer0 = state[ig - targets[0] - targets[1] - targets[2] - targets[3] - targets[4]];
  const T buffer1 = state[ig - targets[1] - targets[2] - targets[3] - targets[4]];
  const T buffer2 = state[ig - targets[0] - targets[2] - targets[3] - targets[4]];
  const T buffer3 = state[ig - targets[2] - targets[3] - targets[4]];
  const T buffer4 = state[ig - targets[0] - targets[1] - targets[3] - targets[4]];
  const T buffer5 = state[ig - targets[1] - targets[3] - targets[4]];
  const T buffer6 = state[ig - targets[0] - targets[3] - targets[4]];
  const T buffer7 = state[ig - targets[3] - targets[4]];
  const T buffer8 = state[ig - targets[0] - targets[1] - targets[2] - targets[4]];
  const T buffer9 = state[ig - targets[1] - targets[2] - targets[4]];
  const T buffer10 = state[ig - targets[0] - targets[2] - targets[4]];
  const T buffer11 = state[ig - targets[2] - targets[4]];
  const T buffer12 = state[ig - targets[0] - targets[1] - targets[4]];
  const T buffer13 = state[ig - targets[1] - targets[4]];
  const T buffer14 = state[ig - targets[0] - targets[4]];
  const T buffer15 = state[ig - targets[4]];
  const T buffer16 = state[ig - targets[0] - targets[1] - targets[2] - targets[3]];
  const T buffer17 = state[ig - targets[1] - targets[2] - targets[3]];
  const T buffer18 = state[ig - targets[0] - targets[2] - targets[3]];
  const T buffer19 = state[ig - targets[2] - targets[3]];
  const T buffer20 = state[ig - targets[0] - targets[1] - targets[3]];
  const T buffer21 = state[ig - targets[1] - targets[3]];
  const T buffer22 = state[ig - targets[0] - targets[3]];
  const T buffer23 = state[ig - targets[3]];
  const T buffer24 = state[ig - targets[0] - targets[1] - targets[2]];
  const T buffer25 = state[ig - targets[1] - targets[2]];
  const T buffer26 = state[ig - targets[0] - targets[2]];
  const T buffer27 = state[ig - targets[2]];
  const T buffer28 = state[ig - targets[0] - targets[1]];
  const T buffer29 = state[ig - targets[1]];
  const T buffer30 = state[ig - targets[0]];
  const T buffer31 = state[ig];

  // Apply the gate to state vector
  // Gate here is a 32x32 matrix
  // Iterate on the 32 output elements indexed by t
  for(auto i = 0; i < 32; i++) {
    // Get the index where the state vector must be updated
    const long t = ig - multitarget_index(targets, 31 - i, ntargets);

    // Update the state vector at index t
    state[t] = gate[i*32]    * buffer0  + gate[i*32+1]  * buffer1  + gate[i*32+2]  * buffer2
             + gate[i*32+3]  * buffer3  + gate[i*32+4]  * buffer4  + gate[i*32+5]  * buffer5
             + gate[i*32+6]  * buffer6  + gate[i*32+7]  * buffer7  + gate[i*32+8]  * buffer8
             + gate[i*32+9]  * buffer9  + gate[i*32+10] * buffer10 + gate[i*32+11] * buffer11
             + gate[i*32+12] * buffer12 + gate[i*32+13] * buffer13 + gate[i*32+14] * buffer14
             + gate[i*32+15] * buffer15 + gate[i*32+16] * buffer16 + gate[i*32+17] * buffer17
             + gate[i*32+18] * buffer18 + gate[i*32+19] * buffer19 + gate[i*32+20] * buffer20
             + gate[i*32+21] * buffer21 + gate[i*32+22] * buffer22 + gate[i*32+23] * buffer23
             + gate[i*32+24] * buffer24 + gate[i*32+25] * buffer25 + gate[i*32+26] * buffer26
             + gate[i*32+27] * buffer27 + gate[i*32+28] * buffer28 + gate[i*32+29] * buffer29
             + gate[i*32+30] * buffer30 + gate[i*32+31] * buffer31;
  }
}


template<typename T>
__global__ void apply_multi_qubit_gate_kernel(T* state, T* buffer,
                                              const T* gate,
                                              const int* qubits,
                                              const long* targets,
                                              long nsubstates,
                                              int ntargets,
                                              int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long ig = multicontrol_index(qubits, g, ncontrols);
  for (auto i = 0; i < nsubstates; i++) {
    const long t = ig - multitarget_index(targets, nsubstates - i - 1, ntargets);
    state[t] = T(0., 0.);
    for (auto j = 0; j < nsubstates; j++) {
      const long u = ig - multitarget_index(targets, nsubstates - j - 1, ntargets);
      state[t] = cadd(state[t], cmult(gate[nsubstates * i + j], buffer[u]));
    }
  }
}


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


template <typename T>
__global__ void initial_state_kernel(T* state) {
  state[0] = T(1, 0);
}
