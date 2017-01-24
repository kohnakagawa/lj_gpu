#pragma once

__device__ __forceinline__ int lane_id() {
  return threadIdx.x % warpSize;
}

template <typename T>
__device__ __forceinline__ T warp_segment_reduce(T var) {
  for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    var += __shfl_down(var, offset);
  }
  return var;
}

#if __CUDACC_VER_MAJOR__ < 8
__device__ __forceinline__ double atomicAdd(double* address, double val) {
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
