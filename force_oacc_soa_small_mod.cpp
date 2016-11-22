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
// const int D = 4;
// enum {X, Y, Z};
//double q[N][D];
//double p[N][D];
double qx[N], qy[N], qz[N];
double px[N], py[N], pz[N];

int particle_number = 0;
int number_of_pairs = 0;
int number_of_partners[N];
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int pointer[N], pointer2[N];
int sorted_list[MAX_PAIRS];

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
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
  qx[particle_number] = x + uniform();
  qy[particle_number] = y + uniform();
  qz[particle_number] = z + uniform();
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
      const double dx = qx[i] - qx[j];
      const double dy = qy[i] - qy[j];
      const double dz = qz[i] - qz[j];
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
    px[i] = 0.0;
    py[i] = 0.0;
    pz[i] = 0.0;
  }
}
//----------------------------------------------------------------------
void
force_sorted(const double* __restrict qx,
             const double* __restrict qy,
             const double* __restrict qz,
             double* __restrict px,
             double* __restrict py,
             double* __restrict pz,
             const int* __restrict number_of_partners,
             const int* __restrict pointer,
             const int* __restrict sorted_list){
  const int pn =particle_number;
#pragma acc kernels present(qx, qy, qz, px, py, pz, number_of_partners, pointer, sorted_list)
  for (int i=0; i<pn; i++) {
    const double qx_key = qx[i];
    const double qy_key = qy[i];
    const double qz_key = qz[i];
    const int np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    for (int k=0; k<np; k++) {
      const int j = sorted_list[kp + k];
      double dx = qx[j] - qx_key;
      double dy = qy[j] - qy_key;
      double dz = qz[j] - qz_key;
      double r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
#pragma acc atomic update
        px[j] -= df*dx;
#pragma acc atomic update
        py[j] -= df*dy;
#pragma acc atomic update
        pz[j] -= df*dz;
    }
#pragma acc atomic update
      px[i] += pfx;
#pragma acc atomic update
      py[i] += pfy;
#pragma acc atomic update
      pz[i] += pfz;
  }
}
//----------------------------------------------------------------------
void
force_reactless(const double* __restrict qx,
                const double* __restrict qy,
                const double* __restrict qz,
                double* __restrict px,
                double* __restrict py,
                double* __restrict pz,
                const int* __restrict number_of_partners,
                const int* __restrict pointer,
                const int* __restrict sorted_list){
  const int pn = particle_number;
#pragma acc kernels present(qx, qy, qz, px, py, pz, number_of_partners, pointer, sorted_list)
  for (int i=0; i<pn; i++) {
    const double qx_key = qx[i];
    const double qy_key = qy[i];
    const double qz_key = qz[i];
    const int np = number_of_partners[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    for (int k=0; k<np; k++) {
      const int j = sorted_list[kp + k];
      double dx = qx[j] - qx_key;
      double dy = qy[j] - qy_key;
      double dz = qz[j] - qz_key;
      double r2 = (dx*dx + dy*dy + dz*dz);
      double r6 = r2*r2*r2;
      double df = ((24.0*r6-48.0)/(r6*r6*r2))*dt;
      if (r2 > CL2) df = 0.0;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
    }
    px[i] += pfx;
    py[i] += pfy;
    pz[i] += pfz;
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
template <typename ptr_func>
void
measure_gpu(ptr_func kernel, const char *name) {
  double st = myclock();
  const int LOOP = 100;
#pragma acc data copy(px[0:N], py[0:N], pz[0:N]) copyin(qx[0:N], qy[0:N], qz[0:N], number_of_partners[0:N], pointer[0:N], sorted_list[0:MAX_PAIRS])
  {
    for (int i = 0; i < LOOP; i++) {
      kernel(qx, qy, qz, px, py, pz, number_of_partners, pointer, sorted_list);
    }
  }
  double t = myclock() - st;
  fprintf(stderr, "N=%d, %s %f [sec]\n", particle_number, name, t);
}
//----------------------------------------------------------------------
int
main(void) {
  init();
  makepair();
  random_shfl();
#ifdef EN_ACTION_REACTION
  measure_gpu(&force_sorted, "acc_sorted");
  for (int i = 0; i < 10; i++) {
    printf("%.10f %.10f %.10f\n", px[i], py[i], pz[i]);
  }
#elif OACC
  measure_gpu(&force_reactless, "acc_reactless_soa_min");
  for (int i = 0; i < 10; i++) {
    printf("%.10f %.10f %.10f\n", px[i], py[i], pz[i]);
  }
#endif
}
//----------------------------------------------------------------------
