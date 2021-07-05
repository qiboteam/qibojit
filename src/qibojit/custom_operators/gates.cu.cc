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
