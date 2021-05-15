#include <cupy/complex.cuh>

__device__ complex<double> cmult(complex<double> a, complex<double> b) {
  return complex<double>(a.real() * b.real() - a.imag() * b.imag(),
                         a.real() * b.imag() + a.imag() * b.real());
}

__device__ complex<double> cadd(complex<double> a, complex<double> b) {
  return complex<double>(a.real() + b.real(), a.imag() + b.imag());
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


__device__ void _apply_gate(complex<double>& state1, complex<double>& state2,
                            const complex<double>* gate) {
  const complex<double> buffer = state1;
  state1 = cadd(cmult(gate[0], state1), cmult(gate[1], state2));
  state2 = cadd(cmult(gate[2], buffer), cmult(gate[3], state2));
}

__device__ void _apply_x(complex<double>& state1, complex<double>& state2) {
  const complex<double> buffer = state1;
  state1 = state2;
  state2 = buffer;
}

__device__ void _apply_y(complex<double>& state1, complex<double>& state2) {
  state1 = cmult(state1, complex<double>(0, 1));
  state2 = cmult(state2, complex<double>(0, -1));
  const complex<double> buffer = state1;
  state1 = state2;
  state2 = buffer;
}

__device__ void _apply_z(complex<double>& state) {
  state = cmult(state, complex<double>(-1));
}

__device__ void _apply_z_pow(complex<double>& state, complex<double> gate) {
  state = cmult(state, gate);
}

__device__ void _apply_two_qubit_gate(complex<double>& state0, complex<double>& state1,
                                      complex<double>& state2, complex<double>& state3,
                                      const complex<double>* gate) {
  const complex<double> buffer0 = state0;
  const complex<double> buffer1 = state1;
  const complex<double> buffer2 = state2;
  state0 = cadd(cadd(cmult(gate[0], state0), cmult(gate[1], state1)),
                cadd(cmult(gate[2], state2), cmult(gate[3], state3)));
  state1 = cadd(cadd(cmult(gate[4], buffer0), cmult(gate[5], state1)),
                cadd(cmult(gate[6], state2), cmult(gate[7], state3)));
  state2 = cadd(cadd(cmult(gate[8], buffer0), cmult(gate[9], buffer1)),
                cadd(cmult(gate[10], state2), cmult(gate[11], state3)));
  state3 = cadd(cadd(cmult(gate[12], buffer0), cmult(gate[13], buffer1)),
                cadd(cmult(gate[14], buffer2), cmult(gate[15], state3)));
}


extern "C" {

__global__ void apply_gate_kernel(complex<double>* state, long tk, int m,
                                  const complex<double>* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_gate(state[i], state[i + tk], gate);
}

__global__ void apply_x_kernel(complex<double>* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_x(state[i], state[i + tk]);
}

__global__ void apply_y_kernel(complex<double>* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_y(state[i], state[i + tk]);
}

__global__ void apply_z_kernel(complex<double>* state, long tk, int m) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z(state[i + tk]);
}

__global__ void apply_z_pow_kernel(complex<double>* state, long tk, int m,
                                   const complex<double>* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = ((long)((long)g >> m) << (m + 1)) + (g & (tk - 1));
  _apply_z_pow(state[i + tk], gate[0]);
}

__global__ void apply_two_qubit_gate_kernel(complex<double>* state, long tk1, long tk2,
                                            int m1, int m2, long uk1, long uk2,
                                            const complex<double>* gate) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_two_qubit_gate(state[i], state[i + uk1], state[i + uk2], state[i + uk1 + uk2], gate);
}

__global__ void apply_swap_kernel(complex<double>* state, long tk1, long tk2,
                                  int m1, int m2, long uk1, long uk2) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  long i = ((long)((long)g >> m1) << (m1 + 1)) + (g & (tk1 - 1));
  i = ((long)((long)i >> m2) << (m2 + 1)) + (i & (tk2 - 1));
  _apply_x(state[i + tk2], state[i + tk1]);
}


__global__ void multicontrol_apply_gate_kernel(complex<double>* state, long tk, int m,
                                               const complex<double>* gate,
                                               const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_gate(state[i - tk], state[i], gate);
}

__global__ void multicontrol_apply_x_kernel(complex<double>* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x(state[i - tk], state[i]);
}

__global__ void multicontrol_apply_y_kernel(complex<double>* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_y(state[i - tk], state[i]);
}

__global__ void multicontrol_apply_z_kernel(complex<double>* state, long tk, int m,
                                            const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z(state[i]);
}

__global__ void multicontrol_apply_z_pow_kernel(complex<double>* state, long tk, int m,
                                                const complex<double>* gate,
                                                const int* qubits, int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_z_pow(state[i], gate[0]);
}

__global__ void multicontrol_apply_two_qubit_gate_kernel(complex<double>* state,
                                                         long tk1, long tk2,
                                                         int m1, int m2,
                                                         long uk1, long uk2,
                                                         const complex<double>* gate,
                                                         const int* qubits,
                                                         int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_two_qubit_gate(state[i - uk1 - uk2], state[i - uk2], state[i - uk1], state[i], gate);
}

__global__ void multicontrol_apply_swap_kernel(complex<double>* state,
                                               long tk1, long tk2,
                                               int m1, int m2,
                                               long uk1, long uk2,
                                               const int* qubits,
                                               int ncontrols) {
  const long g = blockIdx.x * blockDim.x + threadIdx.x;
  const long i = multicontrol_index(qubits, g, ncontrols);
  _apply_x(state[i - tk1], state[i - tk2]);
}
}
