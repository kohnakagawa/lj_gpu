#pragma once

#if 1
typedef double4 Vec;
typedef double Dtype;
#else
typedef float4 Vec;
typedef float Dtype;
#endif

#if 0
// use action and reaction
template <typename Dtype>
__global__ void force_kernel_with_aar(const Vec* q,
                                      Vec* p,
                                      const int32_t* sorted_list,
                                      const int32_t* number_of_partners,
                                      const int32_t* pointer,
                                      const int32_t particle_number,
                                      const Dtype dt,
                                      const Dtype CL2) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const auto kp = pointer[tid];

    for (int32_t k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;

      if (r2 <= CL2) {
        const auto r6 = r2 * r2 * r2;
        const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
        const Dtype dfx = df * dx;
        const Dtype dfy = df * dy;
        const Dtype dfz = df * dz;

        atomicAdd(&p[j].x, -dfx);
        atomicAdd(&p[j].y, -dfy);
        atomicAdd(&p[j].z, -dfz);

        atomicAdd(&p[tid].x, dfx);
        atomicAdd(&p[tid].y, dfy);
        atomicAdd(&p[tid].z, dfy);
      }
    }
  }
}
#endif

// without action and reaction
template <typename Dtype>
__global__ void force_kernel_plain(const Vec* q,
                                   Vec* p,
                                   const int32_t* sorted_list,
                                   const int32_t* number_of_partners,
                                   const int32_t* pointer,
                                   const int32_t particle_number,
                                   const Dtype dt,
                                   const Dtype CL2) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const auto kp = pointer[tid];

    for (int32_t k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      p[tid].x += df * dx;
      p[tid].y += df * dy;
      p[tid].z += df * dz;
    }
  }
}

// with memory access opt
template <typename Dtype>
__global__ void force_kernel_memopt(const Vec*   __restrict__ q,
                                    Vec*         __restrict__ p,
                                    const int32_t* __restrict__ sorted_list,
                                    const int32_t* __restrict__ number_of_partners,
                                    const int32_t* __restrict__ pointer,
                                    const int32_t particle_number,
                                    const Dtype dt,
                                    const Dtype CL2) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const auto kp = pointer[tid];

    Dtype pfx = 0.0, pfy = 0.0, pfz = 0.0;
    for (int32_t k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k]; // use Read Only Cache
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
    }
    p[tid].x += pfx;
    p[tid].y += pfy;
    p[tid].z += pfz;
  }
}

// with loop unrolling


// with software pipelining
