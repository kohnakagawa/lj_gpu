#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "ocl_device.hpp"
#include "ocl_buf.hpp"

const char srcStr[] =
  #include "kernel.cl"
  ;

typedef double Dtype;
typedef cl_double3 Vec;

const Dtype density = 1.0;
const int N = 400000;
// const int N = 1000000;
const int NUM_NEIGH = 60;
const int MAX_PAIRS = NUM_NEIGH * N;
const int LOOP = 100;
Dtype L = 50.0;
// Dtype L = 70.0;
const Dtype dt = 0.001;
ocl_buf<Vec> q, p;
ocl_buf<cl_int> sorted_list, number_of_partners, pointer;
ocl_buf<cl_int> aligned_list;
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
  q[particle_number].s[0] = x + ud(mt);
  q[particle_number].s[1] = y + ud(mt);
  q[particle_number].s[2] = z + ud(mt);
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
    p[i].s[0] = 0.0;
    p[i].s[1] = 0.0;
    p[i].s[2] = 0.0;
  }

  if (particle_number > N) {
    fprintf(stderr, "particle_number %d exceeds maximum buffer size %d\n", particle_number, N);
    std::quick_exit(EXIT_FAILURE);
  }
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
  if (number_of_pairs >= MAX_PAIRS) {
    std::cout << number_of_pairs << " " << MAX_PAIRS << "\n";
    std::exit(1);
  }
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
      const auto dx = q[i].s[0] - q[j].s[0];
      const auto dy = q[i].s[1] - q[j].s[1];
      const auto dz = q[i].s[2] - q[j].s[2];
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
  static std::mt19937 mt;
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto kp = pointer[i];
    const auto np = number_of_partners[i];
    std::shuffle(&sorted_list[kp], &sorted_list[kp + np], mt);
  }
}

template <typename Vec1, typename Vec2>
void copy_vec(Vec1* v1,
              Vec2* v2,
              const int ptcl_num) {
  for (int i = 0; i < ptcl_num; i++) {
    v1[i].s[0] = v2[i].s[0];
    v1[i].s[1] = v2[i].s[1];
    v1[i].s[2] = v2[i].s[2];
  }
}

// for check
template <typename Vec>
void force_sorted(const Vec* q,
                  Vec* p) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[i].s[0];
    const auto qy_key = q[i].s[1];
    const auto qz_key = q[i].s[2];
    const auto np = number_of_partners[i];
    Dtype pfx = 0;
    Dtype pfy = 0;
    Dtype pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].s[0] - qx_key;
      const auto dy = q[j].s[1] - qy_key;
      const auto dz = q[j].s[2] - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((static_cast<Dtype>(24.0)*r6-static_cast<Dtype>(48.0))/(r6*r6*r2))*dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
#ifdef EN_ACTION_REACTION
      p[j].s[0] -= df*dx;
      p[j].s[1] -= df*dy;
      p[j].s[2] -= df*dz;
#endif
    }
    p[i].s[0] += pfx;
    p[i].s[1] += pfy;
    p[i].s[2] += pfz;
  }
}

template <typename Vec>
void print_head_momentum(const Vec* p) {
  for (int i = 0; i < 10; i++) {
    fprintf(stdout, "%.10f %.10f %.10f\n", p[i].s[0], p[i].s[1], p[i].s[2]);
  }
}

int main() {
  std::string ocl_fname("force_kernel_plain");
  std::string compile_opt("-DVec=double3 -DDtype=double");

  // initialize device
  OclDevice ocldevice;
  ocldevice.Initialize();
  ocldevice.AddProgramSource(srcStr);
  ocldevice.AddKernelName(ocl_fname);
  ocldevice.CreateContext();
  ocldevice.BuildProgram(compile_opt);
  ocldevice.CreateKernels();
  cl::Context& context = ocldevice.GetCurrentContext();
  cl::Device&  device  = ocldevice.GetCurrentDevice();

  // allocate memory
  q.Allocate(N, context, CL_MEM_READ_ONLY);
  p.Allocate(N, context, CL_MEM_READ_WRITE);
  sorted_list.Allocate(MAX_PAIRS, context, CL_MEM_READ_ONLY);
  aligned_list.Allocate(MAX_PAIRS, context, CL_MEM_READ_ONLY);
  number_of_partners.Allocate(N, context, CL_MEM_READ_ONLY);
  pointer.Allocate(N, context, CL_MEM_READ_ONLY);

  init(&q[0], &p[0]);

  if (!loadpair()) {
    fprintf(stderr, "Now make pairlist %s.\n", cache_file_name);
    number_of_pairs = 0;
    makepair(&q[0]);
    random_shfl();
    makepaircache();
  }

  make_aligned_pairlist();

#ifdef EN_TEST_CPU
  for (int i = 0; i < LOOP; i++) force_sorted(&q[0], &p[0]);
  print_head_momentum(&p[0]);
#else

  ocldevice.SetFunctionArg(ocl_fname, 0, q.GetDevBuffer());
  ocldevice.SetFunctionArg(ocl_fname, 1, p.GetDevBuffer());
  ocldevice.SetFunctionArg(ocl_fname, 2, particle_number);
  ocldevice.SetFunctionArg(ocl_fname, 3, dt);
  ocldevice.SetFunctionArg(ocl_fname, 4, CL2);
  ocldevice.SetFunctionArg(ocl_fname, 5, aligned_list.GetDevBuffer());
  ocldevice.SetFunctionArg(ocl_fname, 6, number_of_partners.GetDevBuffer());
  cl::Kernel& kernel = ocldevice.GetKernel(ocl_fname);

  // calculate on GPU
  const auto st = myclock();

  const cl::CommandQueue queue(context, device);

  // copy from cpu to gpu
  q.Host2dev(queue, CL_FALSE);
  p.Host2dev(queue, CL_FALSE);
  sorted_list.Host2dev(queue, CL_FALSE);
  number_of_partners.Host2dev(queue, CL_FALSE);
  pointer.Host2dev(queue, CL_FALSE);
  aligned_list.Host2dev(queue, CL_FALSE);
  for (int i = 0; i < LOOP; i++) {
    // cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(particle_number), cl::NullRange);
    // event.wait();
  }

  // copy from gpu to cpu
  p.Dev2host(queue, CL_TRUE);
  const auto diff = myclock() - st;
  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, ocl_fname.c_str(), diff);
  //

  print_head_momentum(&p[0]);
#endif
}
