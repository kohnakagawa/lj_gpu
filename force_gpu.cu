#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "cuda_ptr.cuh"
#include "kernel.cuh"

const double density = 1.0;
const int N = 400000;
const int MAX_PAIRS = 60 * N;
double L = 50.0;
const double dt = 0.001;
cuda_ptr<Vec> q, p;
cuda_ptr<int32_t> sorted_list, number_of_partners, pointer;
int particle_number = 0;
int number_of_pairs = 0;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int pointer2[N];

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
const char* cache_file_name = ".cache_pair.dat";

enum {THREAD_BLOCKS = 256};

void add_particle(const double x,
                  const double y,
                  const double z) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<double> ud(0.0, 0.1);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
  q[particle_number].w = 0.0;
  particle_number++;
}

void init() {
  const double s = 1.0 / std::pow(density * 0.25, 1.0 / 3.0);
  const double hs = s * 0.5;
  const int sx = static_cast<int>(L / s);
  const int sy = static_cast<int>(L / s);
  const int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        const double x = ix*s;
        const double y = iy*s;
        const double z = iz*s;
        add_particle(x     ,y   ,z);
        add_particle(x     ,y+hs,z+hs);
        add_particle(x+hs  ,y   ,z+hs);
        add_particle(x+hs  ,y+hs,z);
      }
    }
  }
  for (int i = 0; i < particle_number; i++) {
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
    p[i].w = 0.0;
  }
}

double myclock() {
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + t.tv_usec * 1e-6;
}

void register_pair(const int index1, const int index2) {
  int i, j;
  if (index1 < index2) {
    i = index1;
    j = index2;
  } else {
    i = index2;
    j = index1;
  }
  i_particles[number_of_pairs] = i;
  j_particles[number_of_pairs] = j;
  number_of_partners[i]++;
  number_of_pairs++;
}

void makepair() {
  const auto SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
  for (int i = 0; i < particle_number; i++) {
    for (int j = 0; j < particle_number; j++) {
      if (i == j) continue;
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

void allocate() {
  q.allocate(N);
  p.allocate(N);
  sorted_list.allocate(MAX_PAIRS);
  number_of_partners.allocate(N);
  pointer.allocate(N);
}

void cleanup() {
  q.deallocate();
  p.deallocate();
  sorted_list.deallocate();
  number_of_partners.deallocate();
  pointer.deallocate();
}

void copy_to_gpu() {
  q.host2dev();
  p.host2dev();
  number_of_partners.host2dev();
  pointer.host2dev();
  sorted_list.host2dev();
}

void copy_to_host() {
  p.dev2host();
}

// for check
void force_sorted() {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[i].x;
    const auto qy_key = q[i].y;
    const auto qz_key = q[i].z;
    const auto np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}

int main() {
  allocate();
  init();
  const auto flag = loadpair();
  if (!flag) {
    fprintf(stderr, "Now make pairlist %s.\n", cache_file_name);
    number_of_pairs = 0;
    makepair();
    makepaircache();
  }
  
  const int block_num = particle_number / THREAD_BLOCKS + 1;
  assert(block_num > 0);

  const auto st = myclock();
  copy_to_gpu();
  const int LOOP = 100;
  for (int i = 0; i < LOOP; i++) {

#ifdef EN_TEST
    force_sorted();
#else

    // force_kernel_plain<Dtype><<<THREAD_BLOCKS, block_num>>>(q, p, sorted_list, number_of_partners, pointer, particle_number, dt, CL2);
    
    force_kernel_memopt<Dtype><<<THREAD_BLOCKS, block_num>>>(q, p, sorted_list, number_of_partners, pointer, particle_number, dt, CL2);

#endif
  }
#ifndef EN_TEST
  copy_to_host();
#endif
  const auto diff = myclock() - st;

  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, "plain", diff);

  for (int i = 0; i < 10; i++) {
    fprintf(stdout, "%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }

  cleanup();
}