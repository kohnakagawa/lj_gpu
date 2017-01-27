#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "cuda_ptr.cuh"
#include "kernel.cuh"

typedef double Dtype;

const Dtype density = 1.0;
const int N = 400000;
// const int N = 1000000;
const int NUM_NEIGH = 60;
const int MAX_PAIRS = NUM_NEIGH * N;
const int LOOP = 100;
Dtype L = 50.0;
// Dtype L = 70.0;
const Dtype dt = 0.001;
cuda_ptr<float3> q_f3, p_f3;
cuda_ptr<float4> q_f4, p_f4;
cuda_ptr<double3> q_d3, p_d3;
cuda_ptr<double4> q_d4, p_d4;
cuda_ptr<int32_t> sorted_list, number_of_partners, pointer;
cuda_ptr<int32_t> aligned_list;
int particle_number = 0;
int number_of_pairs = 0;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int pointer2[N];

const Dtype CUTOFF_LENGTH = 3.0;
const Dtype SEARCH_LENGTH = 3.3;
const auto CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
#ifdef EN_ACTION_REACTION
const char* cache_file_name = ".cache_pair_half.dat";
#else
const char* cache_file_name = ".cache_pair_all.dat";
#endif
const int THREAD_BLOCK = 128;

template <typename Vec>
void add_particle(const Dtype x,
                  const Dtype y,
                  const Dtype z,
                  Vec* q) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<Dtype> ud(0.0, 0.1);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
  particle_number++;
}

template <typename Vec>
void init(Vec* q,
          Vec* p) {
  const Dtype s = 1.0 / std::pow(density * 0.25, 1.0 / 3.0);
  const Dtype hs = s * 0.5;
  const int sx = static_cast<int>(L / s);
  const int sy = static_cast<int>(L / s);
  const int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        const Dtype x = ix*s;
        const Dtype y = iy*s;
        const Dtype z = iz*s;
        add_particle(x     ,y   ,z, q);
        add_particle(x     ,y+hs,z+hs, q);
        add_particle(x+hs  ,y   ,z+hs, q);
        add_particle(x+hs  ,y+hs,z, q);
      }
    }
  }
  for (int i = 0; i < particle_number; i++) {
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }

  if (particle_number > N) {
    fprintf(stderr, "particle_number %d exceeds maximum buffer size %d\n", particle_number, N);
    std::quick_exit(EXIT_FAILURE);
  }

  // std::mt19937 mt(123);
  // std::shuffle(q, q + particle_number, mt);
}

double myclock() {
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + t.tv_usec * 1e-6;
}

void register_pair(const int index1, const int index2) {
  int i, j;
#ifdef EN_ACTION_REACTION
  if (index1 < index2) {
    i = index1;
    j = index2;
  } else {
    i = index2;
    j = index1;
  }
#else
  i = index1;
  j = index2;
#endif
  i_particles[number_of_pairs] = i;
  j_particles[number_of_pairs] = j;
  number_of_partners[i]++;
  number_of_pairs++;
}

template <typename Vec>
void makepair(const Vec* q) {
  const auto SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
#ifdef EN_ACTION_REACTION
  for (int i = 0; i < particle_number - 1; i++) {
    for (int j = i + 1; j < particle_number; j++) {
#else
  for (int i = 0; i < particle_number; i++) {
    for (int j = 0; j < particle_number; j++) {
      if (i == j) continue;
#endif
      const auto dx = q[i].x - q[j].x;
      const auto dy = q[i].y - q[j].y;
      const auto dz = q[i].z - q[j].z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < SL2) {
        register_pair(i, j);
      }
    }
  }
  int pos = 0;
  pointer[0] = 0;
  for (int i = 0; i < pn - 1; i++) {
    pos += number_of_partners[i];
    pointer[i + 1] = pos;
  }
  for (int i = 0; i < pn; i++) {
    pointer2[i] = 0;
  }
  const auto s = number_of_pairs;
  for (int k = 0; k < s; k++) {
    const auto i = i_particles[k];
    const auto j = j_particles[k];
    const auto index = pointer[i] + pointer2[i];
    sorted_list[index] = j;
    pointer2[i] ++;
  }
}

void makepaircache() {
  FILE* fp = fopen(cache_file_name, "w");

  fprintf(fp, "%d %d\n", particle_number, number_of_pairs);
  for (int i = 0; i < particle_number; i++) {
    fprintf(fp, "%d %d\n", number_of_partners[i], pointer[i]);
  }
  for (int i = 0; i < number_of_pairs; i++) {
    fprintf(fp, "%d\n", sorted_list[i]);
  }
  fclose(fp);
}

bool file_exist(const char* name) {
  struct stat buffer;
  return (stat(name, &buffer) == 0);
}

bool check_loadedpair() {
  for (int i = 0; i < particle_number; i++) {
    if (number_of_partners[i] < 0 || number_of_partners[i] >= particle_number) {
      fprintf(stderr, "number_of_partners[%d] = %d\n", i, number_of_partners[i]);
      return false;
    }
    if (pointer[i] < 0 || pointer[i] >= number_of_pairs + 1) {
      fprintf(stderr, "pointer[%d] = %d\n", i, pointer[i]);
      return false;
    }
  }
  for (int i = 0; i < number_of_pairs; i++) {
    if (sorted_list[i] < 0 || sorted_list[i] >= particle_number) {
      fprintf(stderr, "number_of_pairs[%d] = %d\n", i, sorted_list[i]);
      return false;
    }
  }
  return true;
}

bool loadpair() {
  if (!file_exist(cache_file_name)) return false;
  FILE* fp = fopen(cache_file_name, "r");

  int ptcl_num_tmp = 0;
  fscanf(fp, "%d %d", &ptcl_num_tmp, &number_of_pairs);

  if (ptcl_num_tmp != particle_number) {
    fprintf(stderr, "Pairlist cache data may be broken.\n");
    return false;
  }

  for (int i = 0; i < particle_number; i++) {
    fscanf(fp, "%d %d", &number_of_partners[i], &pointer[i]);
  }
  for (int i = 0; i < number_of_pairs; i++) {
    fscanf(fp, "%d", &sorted_list[i]);
  }

  if (!check_loadedpair()) return false;

  fclose(fp);
  fprintf(stderr, "%s is successfully loaded.\n", cache_file_name);
  return true;
}

void make_aligned_pairlist() {
  for (int i = 0; i < particle_number; i++) {
    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      aligned_list[i + k * particle_number] = j;
    }
  }
}

void random_shfl() {
  std::mt19937 mt(10);
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto kp = pointer[i];
    const auto np = number_of_partners[i];
    std::shuffle(&sorted_list[kp], &sorted_list[kp + np], mt);
  }
}

void allocate() {
  q_f3.allocate(N); p_f3.allocate(N);
  q_d3.allocate(N); p_d3.allocate(N);
  q_f4.allocate(N); p_f4.allocate(N);
  q_d4.allocate(N); p_d4.allocate(N);

  sorted_list.allocate(MAX_PAIRS);
  aligned_list.allocate(MAX_PAIRS);
  number_of_partners.allocate(N);
  pointer.allocate(N);
}

void cleanup() {
  q_f3.deallocate(); p_f3.deallocate();
  q_d3.deallocate(); p_d3.deallocate();
  q_f4.deallocate(); p_f4.deallocate();
  q_d4.deallocate(); p_d4.deallocate();

  sorted_list.deallocate();
  aligned_list.deallocate();
  number_of_partners.deallocate();
  pointer.deallocate();
}

template <typename Vec1, typename Vec2>
void copy_vec(Vec1* v1,
              Vec2* v2,
              const int ptcl_num) {
  for (int i = 0; i < ptcl_num; i++) {
    v1[i].x = v2[i].x;
    v1[i].y = v2[i].y;
    v1[i].z = v2[i].z;
  }
}

template <typename Vec>
void copy_to_gpu(cuda_ptr<Vec>& q,
                 cuda_ptr<Vec>& p) {
  q.host2dev();
  p.host2dev();
  sorted_list.host2dev();
  aligned_list.host2dev();
  number_of_partners.host2dev();
  pointer.host2dev();
}

template <typename Vec>
void copy_to_host(cuda_ptr<Vec>& p) {
  p.dev2host();
}

// for check
template <typename Vec>
void force_sorted(const Vec* q,
                  Vec* p) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[i].x;
    const auto qy_key = q[i].y;
    const auto qz_key = q[i].z;
    const auto np = number_of_partners[i];
    Dtype pfx = 0;
    Dtype pfy = 0;
    Dtype pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((static_cast<Dtype>(24.0)*r6-static_cast<Dtype>(48.0))/(r6*r6*r2))*dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
#ifdef EN_ACTION_REACTION
      p[j].x -= df*dx;
      p[j].y -= df*dy;
      p[j].z -= df*dz;
#endif
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}

template <typename Vec, typename Dtype, typename ptr_func>
void measure(ptr_func kernel,
             const char* name,
             cuda_ptr<Vec>& q,
             cuda_ptr<Vec>& p,
             const Dtype dt_,
             const Dtype CL2_,
             const int32_t* list,
             const int32_t* partner_pointer,
             const int32_t tot_thread) {
  const int block_num = (tot_thread - 1) / THREAD_BLOCK + 1;
  const auto st_all = myclock();
  copy_to_gpu(q, p);
  const auto st_calc = myclock();
  for (int i = 0; i < LOOP; i++) {
    kernel<<<block_num, THREAD_BLOCK>>>(q, p, particle_number, dt_, CL2_, list, number_of_partners, partner_pointer);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  const auto diff_calc = myclock() - st_calc;
  copy_to_host(p);
  const auto diff_all = myclock() - st_all;
  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, name, diff_all);
  fprintf(stderr, "N=%d, %s %f [sec] (without Host<->Device)\n", particle_number, name, diff_calc);
}

template <typename Vec>
void print_head_momentum(const Vec* p) {
  for (int i = 0; i < 10; i++) {
    fprintf(stdout, "%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
}

#define STR(s) #s
#define MEASURE_FOR_ALLTYPES(fname, list, p_pointer, tot_thread) \
  do {                                                \
    measure(fname<float3, float>,                     \
            STR(fname ## _float3),                    \
            q_f3,                                     \
            p_f3,                                     \
            static_cast<float>(dt),                   \
            static_cast<float>(CL2),                  \
            list,                                     \
            p_pointer,                                \
            tot_thread);                              \
    measure(fname<float4, float>,                     \
            STR(fname ## _float4),                    \
            q_f4,                                     \
            p_f4,                                     \
            static_cast<float>(dt),                   \
            static_cast<float>(CL2),                  \
            list,                                     \
            p_pointer,                                \
            tot_thread);                              \
    measure(fname<double3, double>,                   \
            STR(fname ## _double3),                   \
            q_d3,                                     \
            p_d3,                                     \
            static_cast<double>(dt),                  \
            static_cast<double>(CL2),                 \
            list,                                     \
            p_pointer,                                \
            tot_thread);                              \
    measure(fname<double4, double>,                   \
            STR(fname ## _double4),                   \
            q_d4,                                     \
            p_d4,                                     \
            static_cast<double>(dt),                  \
            static_cast<double>(CL2),                 \
            list,                                     \
            p_pointer,                                \
            tot_thread);                              \
  } while (false)

int main() {
  allocate();
  init(&q_d3[0], &p_d3[0]);
  copy_vec(&q_f3[0], &q_d3[0], particle_number); copy_vec(&p_f3[0], &p_d3[0], particle_number);
  copy_vec(&q_f4[0], &q_d3[0], particle_number); copy_vec(&p_f4[0], &p_d3[0], particle_number);
  copy_vec(&q_d4[0], &q_d3[0], particle_number); copy_vec(&p_d4[0], &p_d3[0], particle_number);

  const auto flag = loadpair();
  if (!flag) {
    fprintf(stderr, "Now make pairlist %s.\n", cache_file_name);
    number_of_pairs = 0;
    makepair(&q_d3[0]);
    random_shfl();
    makepaircache();
  }

  make_aligned_pairlist();

#ifdef EN_TEST_CPU
  for (int i = 0; i < LOOP; i++) force_sorted(&q_d3[0], &p_d3[0]);
  print_head_momentum(&p_d3[0]);
#elif defined EN_TEST_GPU
  // MEASURE_FOR_ALLTYPES(force_kernel_plain, sorted_list, pointer, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_ifless, sorted_list, pointer, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_memopt, sorted_list, pointer, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_memopt2, sorted_list, pointer, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_swpl, aligned_list, nullptr, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_swpl2, aligned_list, nullptr, particle_number);
  // MEASURE_FOR_ALLTYPES(force_kernel_unrolling, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_warp_unroll, sorted_list, pointer, particle_number * 32);
  print_head_momentum(&p_d3[0]);
#elif defined EN_ACTION_REACTION
  MEASURE_FOR_ALLTYPES(force_kernel_plain_with_aar, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_ifless_with_aar, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_memopt_with_aar, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_memopt2_with_aar, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_warp_unroll_with_aar, sorted_list, pointer, particle_number);
  // print_head_momentum(&p_d3[0]);
#else
  MEASURE_FOR_ALLTYPES(force_kernel_plain, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_ifless, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_memopt, sorted_list, pointer, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_memopt2, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_swpl, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_swpl2, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_unrolling, aligned_list, nullptr, particle_number);
  MEASURE_FOR_ALLTYPES(force_kernel_warp_unroll, sorted_list, pointer, particle_number * 32);
#endif

  cleanup();
}
