#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <boost/random.hpp> // instead of C++11 random
#include <sys/time.h>
//----------------------------------------------------------------------
const double density = 1.0;
const int N = 400000;
const int MAX_PAIRS = 60 * N;
double L = 50.0;
const double dt = 0.001;

struct Vec {
#ifdef USE_DOUBLE4
  double x, y, z, w;
#else
  double x, y, z;
#endif
};

Vec* __restrict q = NULL;
Vec* __restrict p = NULL;

int particle_number = 0;
int number_of_pairs = 0;
int* __restrict number_of_partners = NULL;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int pointer2[N];
int* __restrict pointer = NULL;
int* __restrict sorted_list = NULL;
int* __restrict aligned_list = NULL;

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
void
allocate(void) {
  q = new Vec [N];
  p = new Vec [N];

  number_of_partners = new int [N];
  pointer = new int [N];
  sorted_list = new int [MAX_PAIRS];
  aligned_list = new int [MAX_PAIRS];
}
//----------------------------------------------------------------------
void
deallocate(void) {
  delete [] q;
  delete [] p;

  delete [] number_of_partners;
  delete [] pointer;
  delete [] sorted_list;
  delete [] aligned_list;
}
//----------------------------------------------------------------------
double
uniform(void) {
  static boost::random::mt19937 mt(2);
  boost::random::uniform_real_distribution<double> ud(0.0, 0.1);
  const double tmp = ud(mt); // do not use.
  return ud(mt);
}
//----------------------------------------------------------------------
void
add_particle(double x, double y, double z) {
  q[particle_number].x = x + uniform();
  q[particle_number].y = y + uniform();
  q[particle_number].z = z + uniform();
  particle_number++;
}
//----------------------------------------------------------------------
double
myclock(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}
//----------------------------------------------------------------------
void
register_pair(int index1, int index2) {
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
//----------------------------------------------------------------------
void
makepair(void) {
  const double SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const int pn = particle_number;
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
      const double dx = q[i].x - q[j].x;
      const double dy = q[i].y - q[j].y;
      const double dz = q[i].z - q[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
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
  const int s = number_of_pairs;
  for (int k = 0; k < s; k++) {
    int i = i_particles[k];
    int j = j_particles[k];
    int index = pointer[i] + pointer2[i];
    sorted_list[index] = j;
    pointer2[i] ++;
  }
}
//----------------------------------------------------------------------
void
make_aligned_pairlist(void) {
  for (int i = 0; i < particle_number; i++) {
    const int np = number_of_partners[i];
    const int kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const int j = sorted_list[kp + k];
      aligned_list[i + k * particle_number] = j;
    }
  }
}
//----------------------------------------------------------------------
void
random_shfl(void) {
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const int kp = pointer[i];
    const int np = number_of_partners[i];
    std::random_shuffle(&sorted_list[kp], &sorted_list[kp + np]);
  }
}
//----------------------------------------------------------------------
void
init(void) {
  const double s = 1.0 / pow(density * 0.25, 1.0 / 3.0);
  const double hs = s * 0.5;
  int sx = static_cast<int>(L / s);
  int sy = static_cast<int>(L / s);
  int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        double x = ix*s;
        double y = iy*s;
        double z = iz*s;
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
  }
}
//----------------------------------------------------------------------
void
force_sorted(void){
  const int pn =particle_number;
  for (int i=0; i<pn; i++) {
    const double qx_key = q[i].x;
    const double qy_key = q[i].y;
    const double qz_key = q[i].z;
    const int np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    for (int k=0; k<np; k++) {
      const int j = sorted_list[kp + k];
      double dx = q[j].x - qx_key;
      double dy = q[j].y - qy_key;
      double dz = q[j].z - qz_key;
      double r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
      p[j].x -= df*dx;
      p[j].y -= df*dy;
      p[j].z -= df*dz;
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}
//----------------------------------------------------------------------
void
force_reactless(){
  const int pn = particle_number;
#pragma acc kernels present(q, p, number_of_partners, pointer, sorted_list)
  for (int i=0; i<pn; i++) {
    const double qx_key = q[i].x;
    const double qy_key = q[i].y;
    const double qz_key = q[i].z;
    const int np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    for (int k=0; k<np; k++) {
      const int j = sorted_list[kp + k];
      double dx = q[j].x - qx_key;
      double dy = q[j].y - qy_key;
      double dz = q[j].z - qz_key;
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      if (r2 > CL2) df = 0.0;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}
//----------------------------------------------------------------------
void
force_reactless_memopt(){
  const int pn = particle_number;
#pragma acc kernels present(q, p, number_of_partners, aligned_list)
  for (int i=0; i<pn; i++) {
    const double qx_key = q[i].x;
    const double qy_key = q[i].y;
    const double qz_key = q[i].z;
    const int np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    for (int k=0; k<np; k++) {
      const int j = aligned_list[i + k * particle_number];
      double dx = q[j].x - qx_key;
      double dy = q[j].y - qy_key;
      double dz = q[j].z - qz_key;
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      if (r2 > CL2) df = 0.0;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}
//----------------------------------------------------------------------
void
force_reactless_memopt_tuned(){
  const int pn = particle_number;
#pragma acc kernels present(q, p, number_of_partners, aligned_list)
  for (int i=0; i<pn; i++) {
    const Vec qi = q[i];
    const int np = number_of_partners[i];
    double pfx = 0.0;
    double pfy = 0.0;
    double pfz = 0.0;
    for (int k=0; k<np; k++) {
      const int j = aligned_list[i + k * particle_number];
      const double dx = q[j].x - qi.x;
      const double dy = q[j].y - qi.y;
      const double dz = q[j].z - qi.z;
      const double r2 = (dx*dx + dy*dy + dz*dz);
      const double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      if (r2 > CL2) df = 0.0;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
    }
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  }
}
//----------------------------------------------------------------------
void
measure(void(*pfunc)(), const char *name) {
  double st = myclock();
  const int LOOP = 100;
  for (int i = 0; i < LOOP; i++) {
    pfunc();
  }
  double t = myclock() - st;
  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, name, t);
}
//----------------------------------------------------------------------
void
measure_gpu(void(*pfunc)(), const char *name) {
  double st = myclock();
  const int LOOP = 100;
#pragma acc data copy(p[0:N]) copyin(q[0:N], number_of_partners[0:N], pointer[0:N], sorted_list[0:MAX_PAIRS], aligned_list[0:MAX_PAIRS])
  {
    for (int i = 0; i < LOOP; i++) {
      pfunc();
    }
  }
  double t = myclock() - st;
  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, name, t);
}
//----------------------------------------------------------------------
int
main(void) {
  allocate();
  init();
  makepair();
  random_shfl();
  make_aligned_pairlist();
#ifdef EN_ACTION_REACTION
  measure(&force_sorted, "sorted");
  for (int i = 0; i < 10; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
#elif defined OACC
  measure_gpu(&force_reactless, "acc_reactless_aos");
  for (int i = 0; i < 10; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
#elif defined OACC_MEMOPT
  measure_gpu(&force_reactless_memopt_tuned, "acc_reactless_memopt_aos");
  for (int i = 0; i < 10; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
#endif
  deallocate();
}
//----------------------------------------------------------------------
