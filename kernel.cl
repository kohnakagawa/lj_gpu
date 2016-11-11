#define OCL_EXTERNAL_INCLUDE(x) #x
OCL_EXTERNAL_INCLUDE(
__kernel void force_kernel_memopt2(__global const double4* q,
                                   __global double4* p,
                                   const int particle_number,
                                   const double dt,
                                   const double CL2,
                                   __global const int* aligned_list,
                                   __global const int* number_of_partners) {
  const int tid = get_global_id(0);
  if (tid < particle_number) {
    const double4 qi = q[tid];
    const int np = number_of_partners[tid];
    const __global int* ptr_list = &aligned_list[tid];

    double4 pf = p[tid];
    for (int k = 0; k < np; k++) {
      const int j = *ptr_list;
      const double dx = q[j].x - qi.x;
      const double dy = q[j].y - qi.y;
      const double dz = q[j].z - qi.z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r6 = r2 * r2 * r2;
      double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df = 0.0;
      pf.x += df * dx;
      pf.y += df * dy;
      pf.z += df * dz;
      ptr_list += particle_number;
    }
    p[tid] = pf;
  }
}
)
