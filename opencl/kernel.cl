#define OCL_EXTERNAL_INCLUDE(x) #x
OCL_EXTERNAL_INCLUDE(

__kernel void force_kernel_plain(__global const Vec* restrict q,
                                 __global Vec* restrict p,
                                 const int particle_number,
                                 const Dtype dt,
                                 const Dtype CL2,
                                 __global const int* restrict aligned_list,
                                 __global const int* restrict number_of_partners) {
  const uint tid = get_global_id(0);
  if (tid < particle_number) {
    const Vec qi = q[tid];
    const int np = number_of_partners[tid];
    const __global int* ptr_list = &aligned_list[tid];

    Vec pf = p[tid];
    for (int k = 0; k < np; k++) {
      const int j = *ptr_list;
      const Dtype dx = q[j].x - qi.x;
      const Dtype dy = q[j].y - qi.y;
      const Dtype dz = q[j].z - qi.z;
      const Dtype r2 = dx * dx + dy * dy + dz * dz;
      const Dtype r6 = r2 * r2 * r2;
      Dtype df = (((Dtype)24.0 * r6 - (Dtype)48.0) / (r6 * r6 * r2)) * dt;
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
