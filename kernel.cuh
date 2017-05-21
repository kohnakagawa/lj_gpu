#pragma once

#include "device_util.cuh"

template <typename Vec, typename Dtype>
__global__ void force_kernel_plain(const Vec* q,
                                   Vec* p,
                                   const int32_t particle_number,
                                   const Dtype dt,
                                   const Dtype CL2,
                                   const int32_t* sorted_list,
                                   const int32_t* number_of_partners,
                                   const int32_t* pointer) {
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
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      p[tid].x += df * dx;
      p[tid].y += df * dy;
      p[tid].z += df * dz;
    }
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_ifless(const Vec* q,
                                    Vec* p,
                                    const int32_t particle_number,
                                    const Dtype dt,
                                    const Dtype CL2,
                                    const int32_t* sorted_list,
                                    const int32_t* number_of_partners,
                                    const int32_t* pointer) {
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

template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt(const Vec*   __restrict__ q,
                                    Vec*         __restrict__ p,
                                    const int32_t particle_number,
                                    const Dtype dt,
                                    const Dtype CL2,
                                    const int32_t* __restrict__ sorted_list,
                                    const int32_t* __restrict__ number_of_partners,
                                    const int32_t* __restrict__ pointer) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const auto kp = pointer[tid];

    Dtype pfx = 0.0, pfy = 0.0, pfz = 0.0;
    for (int32_t k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
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

template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt2(const Vec*     __restrict__ q,
                                     Vec*           __restrict__ p,
                                     const int32_t particle_number,
                                     const Dtype dt,
                                     const Dtype CL2,
                                     const int32_t* __restrict__ aligned_list,
                                     const int32_t* __restrict__ number_of_partners,
                                     const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];

    auto pf = p[tid];
    for (int32_t k = 0; k < np; k++) {
      const auto j = __ldg(ptr_list); // use ROC
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
      ptr_list += particle_number;
    }
    p[tid] = pf;
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt3(const Vec*     __restrict__ q,
                                     Vec*           __restrict__ p,
                                     const int32_t particle_number,
                                     const Dtype dt,
                                     const Dtype CL2,
                                     const int32_t* __restrict__ aligned_list,
                                     const int32_t* __restrict__ number_of_partners,
                                     const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];

    auto pf = p[tid];
    for (int32_t k = 0; k < np; k++) {
      const auto j = __ldg(ptr_list); // use ROC
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      const auto r14 = r6 * r6 * r2;
      const auto invr14 = 1.0 / r14;
      const auto df_numera = static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0);
      auto df = df_numera * invr14 * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
      ptr_list += particle_number;
    }
    p[tid] = pf;
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_plain_with_aar(const Vec* q,
                                            Vec* p,
                                            const int32_t particle_number,
                                            const Dtype dt,
                                            const Dtype CL2,
                                            const int32_t* sorted_list,
                                            const int32_t* number_of_partners,
                                            const int32_t* pointer) {
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
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      atomicAdd(&p[tid].x, df * dx);
      atomicAdd(&p[tid].y, df * dy);
      atomicAdd(&p[tid].z, df * dz);
      atomicAdd(&p[j].x, -df * dx);
      atomicAdd(&p[j].y, -df * dy);
      atomicAdd(&p[j].z, -df * dz);
    }
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_ifless_with_aar(const Vec* q,
                                             Vec* p,
                                             const int32_t particle_number,
                                             const Dtype dt,
                                             const Dtype CL2,
                                             const int32_t* sorted_list,
                                             const int32_t* number_of_partners,
                                             const int32_t* pointer) {
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
      atomicAdd(&p[tid].x, df * dx);
      atomicAdd(&p[tid].y, df * dy);
      atomicAdd(&p[tid].z, df * dz);
      atomicAdd(&p[j].x, -df * dx);
      atomicAdd(&p[j].y, -df * dy);
      atomicAdd(&p[j].z, -df * dz);
    }
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt_with_aar(const Vec*   __restrict__ q,
                                             Vec*         __restrict__ p,
                                             const int32_t particle_number,
                                             const Dtype dt,
                                             const Dtype CL2,
                                             const int32_t* __restrict__ sorted_list,
                                             const int32_t* __restrict__ number_of_partners,
                                             const int32_t* __restrict__ pointer) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const auto kp = pointer[tid];

    Dtype pfx = 0.0, pfy = 0.0, pfz = 0.0;
    for (int32_t k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
      atomicAdd(&p[j].x, -df * dx);
      atomicAdd(&p[j].y, -df * dy);
      atomicAdd(&p[j].z, -df * dz);
    }
    atomicAdd(&p[tid].x, pfx);
    atomicAdd(&p[tid].y, pfy);
    atomicAdd(&p[tid].z, pfz);
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt2_with_aar(const Vec*     __restrict__ q,
                                              Vec*           __restrict__ p,
                                              const int32_t particle_number,
                                              const Dtype dt,
                                              const Dtype CL2,
                                              const int32_t* __restrict__ aligned_list,
                                              const int32_t* __restrict__ number_of_partners,
                                              const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];

    Dtype pfx = 0.0, pfy = 0.0, pfz = 0.0;
    for (int32_t k = 0; k < np; k++) {
      const auto j = __ldg(ptr_list); // use ROC
      ptr_list += particle_number;
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
      atomicAdd(&p[j].x, -df * dx);
      atomicAdd(&p[j].y, -df * dy);
      atomicAdd(&p[j].z, -df * dz);
    }
    atomicAdd(&p[tid].x, pfx);
    atomicAdd(&p[tid].y, pfy);
    atomicAdd(&p[tid].z, pfz);
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_warp_unroll_with_aar(const Vec*     __restrict__ q,
                                                  Vec*           __restrict__ p,
                                                  const int32_t particle_number,
                                                  const Dtype dt,
                                                  const Dtype CL2,
                                                  const int32_t* __restrict__ sorted_list,
                                                  const int32_t* __restrict__ number_of_partners,
                                                  const int32_t* __restrict__ pointer) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id < particle_number) {
    const auto lid = lane_id();
    const auto qi = q[i_ptcl_id];
    const auto np = number_of_partners[i_ptcl_id];
    const auto kp = pointer[i_ptcl_id] + lid;
    const int32_t ini_loop = (np / warpSize) * warpSize;

    Vec pf = {0.0};
    int32_t k = 0;
    for (; k < ini_loop; k += warpSize) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= CL2) {
        const auto r6 = r2 * r2 * r2;
        const auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
        pf.x += df * dx;
        pf.y += df * dy;
        pf.z += df * dz;
        atomicAdd(&p[j].x, -df * dx);
        atomicAdd(&p[j].y, -df * dy);
        atomicAdd(&p[j].z, -df * dz);
      }
    }

    // remaining loop
    if (lid < (np % warpSize)) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= CL2) {
        const auto r6 = r2 * r2 * r2;
        const auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
        pf.x += df * dx;
        pf.y += df * dy;
        pf.z += df * dz;
        atomicAdd(&p[j].x, -df * dx);
        atomicAdd(&p[j].y, -df * dy);
        atomicAdd(&p[j].z, -df * dz);
      }
    }

    pf.x = warp_segment_reduce(pf.x);
    pf.y = warp_segment_reduce(pf.y);
    pf.z = warp_segment_reduce(pf.z);

    if (lid == 0) {
      atomicAdd(&p[i_ptcl_id].x, pf.x);
      atomicAdd(&p[i_ptcl_id].y, pf.y);
      atomicAdd(&p[i_ptcl_id].z, pf.z);
    }
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_unrolling(const Vec*     __restrict__ q,
                                       Vec*           __restrict__ p,
                                       const int32_t particle_number,
                                       const Dtype dt,
                                       const Dtype CL2,
                                       const int32_t* __restrict__ aligned_list,
                                       const int32_t* __restrict__ number_of_partners,
                                       const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];
    const auto particle_number_x4 = particle_number << 2;

    auto pf = p[tid];
    int32_t k = 0;
    const auto ini_loop = np & 0x3;
    for (; k < ini_loop; k++) {
      const auto j = __ldg(ptr_list);
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
      ptr_list += particle_number;
    }

    for (; k < np; k += 4) {
      const auto j0 = __ldg(ptr_list);
      const auto j1 = __ldg(ptr_list + particle_number);
      const auto j2 = __ldg(ptr_list + 2 * particle_number);
      const auto j3 = __ldg(ptr_list + 3 * particle_number);

      const auto dx0 = q[j0].x - qi.x; const auto dy0 = q[j0].y - qi.y; const auto dz0 = q[j0].z - qi.z;
      const auto dx1 = q[j1].x - qi.x; const auto dy1 = q[j1].y - qi.y; const auto dz1 = q[j1].z - qi.z;
      const auto dx2 = q[j2].x - qi.x; const auto dy2 = q[j2].y - qi.y; const auto dz2 = q[j2].z - qi.z;
      const auto dx3 = q[j3].x - qi.x; const auto dy3 = q[j3].y - qi.y; const auto dz3 = q[j3].z - qi.z;

      const auto r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
      const auto r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
      const auto r2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
      const auto r2_3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

      const auto r6_0 = r2_0 * r2_0 * r2_0;
      const auto r6_1 = r2_1 * r2_1 * r2_1;
      const auto r6_2 = r2_2 * r2_2 * r2_2;
      const auto r6_3 = r2_3 * r2_3 * r2_3;

      auto df0 = ((static_cast<Dtype>(24.0) * r6_0 - static_cast<Dtype>(48.0)) / (r6_0 * r6_0 * r2_0)) * dt;
      auto df1 = ((static_cast<Dtype>(24.0) * r6_1 - static_cast<Dtype>(48.0)) / (r6_1 * r6_1 * r2_1)) * dt;
      auto df2 = ((static_cast<Dtype>(24.0) * r6_2 - static_cast<Dtype>(48.0)) / (r6_2 * r6_2 * r2_2)) * dt;
      auto df3 = ((static_cast<Dtype>(24.0) * r6_3 - static_cast<Dtype>(48.0)) / (r6_3 * r6_3 * r2_3)) * dt;

      if (r2_0 > CL2) df0 = 0.0;
      if (r2_1 > CL2) df1 = 0.0;
      if (r2_2 > CL2) df2 = 0.0;
      if (r2_3 > CL2) df3 = 0.0;

      pf.x += df0 * dx0; pf.x += df1 * dx1; pf.x += df2 * dx2; pf.x += df3 * dx3;
      pf.y += df0 * dy0; pf.y += df1 * dy1; pf.y += df2 * dy2; pf.y += df3 * dy3;
      pf.z += df0 * dz0; pf.z += df1 * dz1; pf.z += df2 * dz2; pf.z += df3 * dz3;

      ptr_list += particle_number_x4;
    }

    p[tid] = pf;
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_unrolling2(const Vec*     __restrict__ q,
                                        Vec*           __restrict__ p,
                                        const int32_t particle_number,
                                        const Dtype dt,
                                        const Dtype CL2,
                                        const int32_t* __restrict__ aligned_list,
                                        const int32_t* __restrict__ number_of_partners,
                                        const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= particle_number) return;

  const auto qi = q[tid];
  const auto np = number_of_partners[tid];
  const int32_t* ptr_list = &aligned_list[tid];
  const auto particle_number_x2 = particle_number << 1;

  auto pf = p[tid];
  int32_t k = np & 0x1;
  if (k) {
    const auto j = __ldg(ptr_list);
    const auto dx = q[j].x - qi.x;
    const auto dy = q[j].y - qi.y;
    const auto dz = q[j].z - qi.z;
    const auto r2 = dx * dx + dy * dy + dz * dz;
    const auto r6 = r2 * r2 * r2;
    const auto r14 = r6 * r6 * r2;
    const auto invr14 = 1.0 / r14;
    const auto df_numera = static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0);
    auto df = df_numera * invr14 * dt;
    if (r2 > CL2) df = 0.0;
    pf.x += df * dx;
    pf.y += df * dy;
    pf.z += df * dz;
    ptr_list += particle_number;
  }

  for (; k < np; k += 2) {
    const auto j0 = __ldg(ptr_list);
    const auto j1 = __ldg(ptr_list + particle_number);

    const auto dx0 = q[j0].x - qi.x; const auto dx1 = q[j1].x - qi.x;
    const auto dy0 = q[j0].y - qi.y; const auto dy1 = q[j1].y - qi.y;
    const auto dz0 = q[j0].z - qi.z; const auto dz1 = q[j1].z - qi.z;

    const auto r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
    const auto r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

    const auto r6_0 = r2_0 * r2_0 * r2_0;
    const auto r6_1 = r2_1 * r2_1 * r2_1;

    const auto r14_0 = r6_0 * r6_0 * r2_0;
    const auto r14_1 = r6_1 * r6_1 * r2_1;

    const auto invr14_01 = 1.0 / (r14_0 * r14_1);

    const auto df_numera_0 = static_cast<Dtype>(24.0) * r6_0 - static_cast<Dtype>(48.0);
    const auto df_numera_1 = static_cast<Dtype>(24.0) * r6_1 - static_cast<Dtype>(48.0);

    auto df0 = df_numera_0 * invr14_01 * r14_1 * dt;
    auto df1 = df_numera_1 * invr14_01 * r14_0 * dt;

    if (r2_0 > CL2) df0 = 0.0;
    if (r2_1 > CL2) df1 = 0.0;

    pf.x += df0 * dx0; pf.x += df1 * dx1;
    pf.y += df0 * dy0; pf.y += df1 * dy1;
    pf.z += df0 * dz0; pf.z += df1 * dz1;

    ptr_list += particle_number_x2;
  }

  p[tid] = pf;
}

// ASSUME: np >= 3
template <typename Vec, typename Dtype>
__global__ void force_kernel_swpl(const Vec*     __restrict__ q,
                                  Vec*           __restrict__ p,
                                  const int32_t particle_number,
                                  const Dtype dt,
                                  const Dtype CL2,
                                  const int32_t* __restrict__ aligned_list,
                                  const int32_t* __restrict__ number_of_partners,
                                  const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];

    auto pf = p[tid];

    Dtype r2, r6, df_0, df_1;

    // load (k == 0)
    auto j = __ldg(ptr_list);
    ptr_list += particle_number;
    auto dx_0 = q[j].x - qi.x;
    auto dy_0 = q[j].y - qi.y;
    auto dz_0 = q[j].z - qi.z;

    // calc (k == 0)
    r2 = dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0;
    r6 = r2 * r2 * r2;
    df_0 = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
    if (r2 > CL2) df_0 = 0.0;

    // load (k == 1)
    j = __ldg(ptr_list);
    ptr_list += particle_number;
    auto dx_1 = q[j].x - qi.x;
    auto dy_1 = q[j].y - qi.y;
    auto dz_1 = q[j].z - qi.z;

    for (int32_t k = 2; k < np; k++) {
      // store (k)
      pf.x += df_0 * dx_0;
      pf.y += df_0 * dy_0;
      pf.z += df_0 * dz_0;

      // calc (k + 1)
      r2 = dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1;
      r6 = r2 * r2 * r2;
      df_0 = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df_0 = 0.0;

      dx_0 = dx_1; dy_0 = dy_1; dz_0 = dz_1;

      // load (k + 2)
      j = __ldg(ptr_list);
      ptr_list += particle_number;
      dx_1 = q[j].x - qi.x;
      dy_1 = q[j].y - qi.y;
      dz_1 = q[j].z - qi.z;
    }

    // calc (k == np - 1)
    r2 = dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1;
    r6 = r2 * r2 * r2;
    df_1 = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
    if (r2 > CL2) df_1 = 0.0;

    // store (k == np - 2, np - 1)
    pf.x += df_1 * dx_1 + df_0 * dx_0;
    pf.y += df_1 * dy_1 + df_0 * dy_0;
    pf.z += df_1 * dz_1 + df_0 * dz_0;

    p[tid] = pf;
 }
}

// modified from lj_simd/force_aos.cpp
template <typename Vec, typename Dtype>
__global__ void force_kernel_swpl2(const Vec*     __restrict__ q,
                                   Vec*           __restrict__ p,
                                   const int32_t particle_number,
                                   const Dtype dt,
                                   const Dtype CL2,
                                   const int32_t* __restrict__ aligned_list,
                                   const int32_t* __restrict__ number_of_partners,
                                   const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto np = number_of_partners[tid];
    const int32_t* ptr_list = &aligned_list[tid];

    auto pf = p[tid];

    auto j = __ldg(ptr_list);
    ptr_list += particle_number;
    auto dxa = q[j].x - qi.x;
    auto dya = q[j].y - qi.y;
    auto dza = q[j].z - qi.z;
    Dtype df = 0.0;
    Dtype dxb = 0.0, dyb = 0.0, dzb = 0.0;

    for (int32_t k = 0; k < np; k++) {
      const auto dx = dxa;
      const auto dy = dya;
      const auto dz = dza;
      const auto r2 = dx * dx + dy * dy + dz * dz;

      j = __ldg(ptr_list);
      ptr_list += particle_number;

      dxa = q[j].x - qi.x;
      dya = q[j].y - qi.y;
      dza = q[j].z - qi.z;

      // store prev
      pf.x += df * dxb;
      pf.y += df * dyb;
      pf.z += df * dzb;

      // calc next
      const auto r6 = r2 * r2 * r2;
      df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      dxb = dx;
      dyb = dy;
      dzb = dz;
    }
    pf.x += df * dxb;
    pf.y += df * dyb;
    pf.z += df * dzb;

    p[tid] = pf;
  }
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_swpl3(const Vec*     __restrict__ q,
                                   Vec*           __restrict__ p,
                                   const int32_t particle_number,
                                   const Dtype dt,
                                   const Dtype CL2,
                                   const int32_t* __restrict__ aligned_list,
                                   const int32_t* __restrict__ number_of_partners,
                                   const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= particle_number) return;

  const auto qi = q[tid];
  const auto np = number_of_partners[tid];
  const int32_t* ptr_list = &aligned_list[tid];

  auto pf = p[tid];

  auto j = __ldg(ptr_list);
  ptr_list += particle_number;
  auto dxa = q[j].x - qi.x;
  auto dya = q[j].y - qi.y;
  auto dza = q[j].z - qi.z;
  Dtype df = 0.0;
  Dtype dxb = 0.0, dyb = 0.0, dzb = 0.0;

  for (int32_t k = 0; k < np; k++) {
    const auto dx = dxa;
    const auto dy = dya;
    const auto dz = dza;
    const auto r2 = dx * dx + dy * dy + dz * dz;

    j = __ldg(ptr_list);
    ptr_list += particle_number;

    dxa = q[j].x - qi.x;
    dya = q[j].y - qi.y;
    dza = q[j].z - qi.z;

    // store prev
    pf.x += df * dxb;
    pf.y += df * dyb;
    pf.z += df * dzb;

    // calc next
    const auto r6 = r2 * r2 * r2;
    const auto r14 = r6 * r6 * r2;
    const auto invr14 = 1.0 / r14;
    const auto df_numera = static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0);
    df = df_numera * invr14 * dt;
    if (r2 > CL2) df = 0.0;
    dxb = dx;
    dyb = dy;
    dzb = dz;
  }
  pf.x += df * dxb;
  pf.y += df * dyb;
  pf.z += df * dzb;

  p[tid] = pf;
}

template <typename Vec, typename Dtype>
__global__ void force_kernel_warp_unroll(const Vec*     __restrict__ q,
                                         Vec*           __restrict__ p,
                                         const int32_t particle_number,
                                         const Dtype dt,
                                         const Dtype CL2,
                                         const int32_t* __restrict__ sorted_list,
                                         const int32_t* __restrict__ number_of_partners,
                                         const int32_t* __restrict__ pointer) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id < particle_number) {
    const auto lid = lane_id();
    const auto qi = q[i_ptcl_id];
    const auto np = number_of_partners[i_ptcl_id];

    Vec pf = {0.0};
    if (lid == 0) pf = p[i_ptcl_id];
    const auto kp = pointer[i_ptcl_id];
    for (int32_t k = lid; k < np; k += warpSize) {
      const auto j  = sorted_list[kp + k];
      const auto dx = q[j].x - qi.x;
      const auto dy = q[j].y - qi.y;
      const auto dz = q[j].z - qi.z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      const auto r6 = r2 * r2 * r2;
      auto df = ((static_cast<Dtype>(24.0) * r6 - static_cast<Dtype>(48.0)) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
    }

    // warp reduction
    pf.x = warp_segment_reduce(pf.x);
    pf.y = warp_segment_reduce(pf.y);
    pf.z = warp_segment_reduce(pf.z);

    if (lid == 0) p[i_ptcl_id] = pf;
  }
}

// ASSUME: particle_number % 2 == 0
template <typename Vec, typename Dtype>
__global__ void force_kernel_memopt3_coarse(const Vec*     __restrict__ q,
                                            Vec*           __restrict__ p,
                                            const int32_t particle_number,
                                            const Dtype dt,
                                            const Dtype CL2,
                                            const int32_t* __restrict__ aligned_list,
                                            const int32_t* __restrict__ number_of_partners,
                                            const int32_t* __restrict__ pointer = nullptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  const auto wid = tid / warpSize;
  const auto lid = tid % warpSize;

  const auto ptcl_id0 = (2 * wid    ) * warpSize + lid;
  const auto ptcl_id1 = (2 * wid + 1) * warpSize + lid;

  const auto qi0 = q[ptcl_id0];
  const auto qi1 = q[ptcl_id1];

  const auto np0 = number_of_partners[ptcl_id0];
  const auto np1 = number_of_partners[ptcl_id1];

  const int32_t* ptr_list0 = &aligned_list[ptcl_id0];
  const int32_t* ptr_list1 = &aligned_list[ptcl_id1];

  const auto np = (np0 > np1) ? np0 : np1;

  auto pf0 = p[ptcl_id0];
  auto pf1 = p[ptcl_id1];
  for (int32_t k = 0; k < np; k++) {
    const auto j0 = *ptr_list0;
    const auto j1 = *ptr_list1;

    const auto dx0 = q[j0].x - qi0.x;
    const auto dy0 = q[j0].y - qi0.y;
    const auto dz0 = q[j0].z - qi0.z;
    const auto dx1 = q[j1].x - qi1.x;
    const auto dy1 = q[j1].y - qi1.y;
    const auto dz1 = q[j1].z - qi1.z;

    const auto r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
    const auto r6_0 = r2_0 * r2_0 * r2_0;
    const auto r14_0 = r6_0 * r6_0 * r2_0;
    const auto invr14_0 = 1.0 / r14_0;
    const auto r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const auto r6_1 = r2_1 * r2_1 * r2_1;
    const auto r14_1 = r6_1 * r6_1 * r2_1;
    const auto invr14_1 = 1.0 / r14_1;

    const auto df0_numera = static_cast<Dtype>(24.0) * r6_0 - static_cast<Dtype>(48.0);
    auto df0 = df0_numera * invr14_0 * dt;
    if (r2_0 > CL2 || k >= np0) df0 = 0.0;
    const auto df1_numera = static_cast<Dtype>(24.0) * r6_1 - static_cast<Dtype>(48.0);
    auto df1 = df1_numera * invr14_1 * dt;
    if (r2_1 > CL2 || k >= np1) df1 = 0.0;

    pf0.x += df0 * dx0;
    pf0.y += df0 * dy0;
    pf0.z += df0 * dz0;
    ptr_list0 += particle_number;
    pf1.x += df1 * dx1;
    pf1.y += df1 * dy1;
    pf1.z += df1 * dz1;
    ptr_list1 += particle_number;
  }
  p[ptcl_id0] = pf0;
  p[ptcl_id1] = pf1;
}
