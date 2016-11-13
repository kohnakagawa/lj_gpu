CACHE_FILE = .cache_pair_half.dat .cache_pair_all.dat

SIMD = aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out
PTX = kernel.ptx kernel_aar.ptx
OCL = gpu_ocl.out cpu_ocl_ref.out
CUDA = gpu_cuda.out gpu_cuda_aar.out gpu_cuda_test.out cpu_cuda_aar_ref.out cpu_cuda_ref.out
OACC = gpu_oacc_soa.out cpu_oacc_ref.out gpu_oacc_aos_d4.out gpu_oacc_aos_d3.out gpu_oacc_aos_memopt_d4.out gpu_oacc_aos_memopt_d3.out

WARNINGS = -Wall -Wextra -Wunused-variable -Wsign-compare
GCC_FLAGS = -O3 -funroll-loops -ffast-math
PGI_FLAGS = -O3
OACC_FLAGS = -acc -ta=nvidia,cc35,keepgpu -Minfo=accel

cuda_profile = yes

AMDAPP_ROOT=/opt/AMDAPPSDK-3.0
BOOST_ROOT=/home/app/boost/1.58

# CUDA_HOME=/usr/local/cuda
CUDA_HOME=/home/app/cuda/cuda-7.0 # for System B
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= -O3 -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(GCC_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc
ifeq ($(cuda_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

simd: $(SIMD)
cuda: $(CUDA) $(PTX)
ocl: $(OCL)
oacc : $(OACC)

aos.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 $< -o $@

aos_pair.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 -DPAIR $< -o $@

aos_intrin.out: force_aos.cpp
	icpc -O3 -xHOST -std=c++11 -DINTRIN $< -o $@

soa.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 $< -o $@

soa_pair.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 -DPAIR $< -o $@

soa_intrin.out: force_soa.cpp
	icpc -O3 -xHOST -std=c++11 -DINTRIN $< -o $@

gpu_cuda.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

gpu_cuda_test.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_GPU $(INCLUDE) $< $(LIBRARY) -o $@

gpu_cuda_aar.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) $< $(LIBRARY) -o $@

cpu_cuda_ref.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_CPU $(INCLUDE) $< $(LIBRARY) -o $@

cpu_cuda_aar_ref.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_CPU -DEN_ACTION_REACTION $(INCLUDE) $< $(LIBRARY) -o $@

cpu_oacc_ref.out: force_oacc_soa.cpp
	pgcpp $(PGI_FLAGS) -DEN_ACTION_REACTION -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

gpu_oacc_soa.out: force_oacc_soa.cpp
	pgcpp $(OACC_FLAGS) $(PGI_FLAGS) -DOACC -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

gpu_oacc_aos_d3.out: force_oacc_aos.cpp
	pgcpp $(OACC_FLAGS) $(PGI_FLAGS) -DOACC -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

gpu_oacc_aos_d4.out: force_oacc_aos.cpp
	pgcpp $(OACC_FLAGS) $(PGI_FLAGS) -DOACC -DUSE_DOUBLE4 -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

gpu_oacc_aos_memopt_d3.out: force_oacc_aos.cpp
	pgcpp $(OACC_FLAGS) $(PGI_FLAGS) -DOACC_MEMOPT -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

gpu_oacc_aos_memopt_d4.out: force_oacc_aos.cpp
	pgcpp $(OACC_FLAGS) $(PGI_FLAGS) -DOACC_MEMOPT -DUSE_DOUBLE4 -I$(BOOST_ROOT)/include $< -L$(BOOST_ROOT)/lib -lboost_system -lboost_random -o $@

cpu_ocl_ref.out: force_ocl.cpp
	g++ -O3 -std=c++11 -DEN_TEST_CPU -I$(AMDAPP_ROOT)/include $< -L$(AMDAPP_ROOT)/lib/x86_64 -lOpenCL -o $@

gpu_ocl.out: force_ocl.cpp
	g++ -O3 -std=c++11 -I$(AMDAPP_ROOT)/include $< -L$(AMDAPP_ROOT)/lib/x86_64 -lOpenCL -o $@

kernel.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

kernel_aar.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) -ptx $< $(LIBRARY) -o $@

clean:
	rm -f $(SIMD) $(CUDA) $(PTX) $(OCL) $(OACC) *.gpu *~ *.core

test_simd: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out
	./aos_pair.out > aos_pair.dat
	./aos_intrin.out > aos_intrin.dat
	diff aos_pair.dat aos_intrin.dat
	./soa_pair.out > soa_pair.dat
	./soa_intrin.out > soa_intrin.dat
	diff soa_pair.dat soa_intrin.dat

test_cuda: cpu_cuda_aar_ref.out cpu_cuda_ref.out gpu_cuda_test.out
	./cpu_cuda_aar_ref.out > cpu_cuda_aar.txt
	./cpu_cuda_ref.out > cpu_cuda_ref.txt
	./gpu_cuda_test.out > gpu_cuda_test.txt
	diff cpu_cuda_aar.txt cpu_cuda_ref.txt
	diff cpu_cuda_ref.txt gpu_cuda_test.txt

test_ocl: $(OCL)
	./cpu_ocl_ref.out > cpu_ocl_ref.txt
	./gpu_ocl.out > gpu_ocl.txt
	diff cpu_ocl_ref.txt gpu_ocl.txt

test_oacc: $(OACC)
	./cpu_oacc_ref.out > cpu_oacc_ref.txt
	./gpu_oacc_soa.out > gpu_oacc_soa.txt
	./gpu_oacc_aos_d4.out > gpu_oacc_aos_d4.txt
	./gpu_oacc_aos_d3.out > gpu_oacc_aos_d3.txt
	./gpu_oacc_aos_memopt_d4.out > gpu_oacc_aos_memopt_d4.txt
	./gpu_oacc_aos_memopt_d3.out > gpu_oacc_aos_memopt_d3.txt
	diff cpu_oacc_ref.txt gpu_oacc_soa.txt
	diff gpu_oacc_soa.txt gpu_oacc_aos_d4.txt
	diff gpu_oacc_aos_d4.txt gpu_oacc_aos_d3.txt
	diff gpu_oacc_aos_d3.txt gpu_oacc_aos_memopt_d4.txt
	diff gpu_oacc_aos_memopt_d4.txt gpu_oacc_aos_memopt_d3.txt

cuda_bench: aos_intrin.out soa_intrin.out aos_pair.out soa_pair.out gpu_cuda_aar.out gpu_cuda.out
	./aos_intrin.out
	./soa_intrin.out
	./aos_pair.out
	./soa_pair.out
	./gpu_cuda_aar.out
	./gpu_cuda.out

clear:
	rm -f $(CACHE_FILE)
