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

// with memory access opt2
__global__ void force_kernel_memopt2(const Vec*     __restrict__ q,
                                     Vec*           __restrict__ p,
                                     const int32_t* __restrict__ aligned_list,
                                     const int32_t* __restrict__ number_of_partners,
                                     const int32_t particle_number,
                                     const Dtype dt,
                                     const Dtype CL2) {
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

// with loop unrolling
__global__ void force_kernel_unrolling(const Vec*     __restrict__ q,
                                       Vec*           __restrict__ p,
                                       const int32_t* __restrict__ aligned_list,
                                       const int32_t* __restrict__ number_of_partners,
                                       const int32_t particle_number,
                                       const Dtype dt,
                                       const Dtype CL2) {
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