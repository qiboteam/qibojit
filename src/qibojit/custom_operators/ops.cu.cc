#include <cupy/complex.cuh>


__device__ long collapse_index(const int* qubits, long g, long h, int ntargets) {
  long i = g;
  for (auto iq = 0; iq < ntargets; iq++) {
    const auto n = qubits[iq];
    long k = (long)1 << n;
    i = ((long)((long)i >> n) << (n + 1)) + (i & (k - 1));
    i += ((long)((int)(h >> iq) % 2) * k);
  }
  return i;
};


template <typename T>
__global__ void collapse_state_kernel(T* state, const int* qubits,
                                      const long* results, int ntargets) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  const long result = results[0];
  const long nsubstates = (long)1 << ntargets;
  for (auto h = 0; h < result; h++) {
    state[collapse_index(qubits, g, h, ntargets)] = T(0, 0);
  }
  for (auto h = result + 1; h < nsubstates; h++) {
    state[collapse_index(qubits, g, h, ntargets)] = T(0, 0);
  }
}


template <typename T, typename R>
__global__ void collapsed_norm_kernel(T* state, const int* qubits,
                                      const long* results, int ntargets,
                                      long nstates, R* norms) {
  const auto tid = threadIdx.x;
  const auto stride = blockDim.x;
  const long result = results[0];
  norms[tid] = 0;
  for (auto g = tid; g < nstates; g += stride) {
    auto x = state[collapse_index(qubits, g, result, ntargets)];
    norms[tid] += x.real() * x.real() + x.imag() * x.imag();
  }
}


template <typename R>
__global__ void vector_reduction_kernel(R *g_idata, R *g_odata) {
  extern __shared__ double sdata[DEFAULT_BLOCK_SIZE];
  // each thread loads one element from global to shared mem
  const auto tid = threadIdx.x;
  sdata[tid] = g_idata[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();
  // do reduction in shared mem
  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = std::sqrt(sdata[0]);
  }
}


template <typename T, typename R>
__global__ void normalize_collapsed_state_kernel(T* state, const int* qubits,
                                                 const long* results,
                                                 int ntargets, long nstates,
                                                 R* norms) {
  const auto g = blockIdx.x * blockDim.x + threadIdx.x;
  auto normalize_component = [&](T& x) {
    x = T(x.real() / norms[0], x.imag() / norms[0]);
  };
  normalize_component(state[collapse_index(qubits, g, results[0], ntargets)]);
}
