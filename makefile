TARGET= aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out
CACHE_FILE = .cache_pair_half.dat .cache_pair_all.dat

PTX = kernel.ptx kernel_aar.ptx
OCL = gpu_ocl.out cpu_ocl_ref.out
CUDA = gpu_cuda.out gpu_cuda_aar.out gpu_cuda_test.out cpu_cuda_aar_ref.out cpu_cuda_ref.out
OACC = gpu_acc.out

WARNINGS = -Wall -Wextra -Wunused-variable -Wsign-compare
OPT_FLAGS = -O3 -funroll-loops -ffast-math

gpu_profile = yes

# CUDA_HOME=/usr/local/cuda
CUDA_HOME=/home/app/cuda/cuda-7.0
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= -O3 -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc
ifeq ($(gpu_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

AMDAPP_ROOT=/opt/AMDAPPSDK-3.0

all: $(TARGET)
cuda: $(CUDA) $(PTX)
ocl: $(OCL)

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

gpu_acc.out: force_acc.cpp
	pgcpp -acc -ta=nvidia -Minfo=accel -fast -O3 -DEN_ACTION_REACTION $< -o $@

cpu_ocl_ref.out: force_ocl.cpp
	g++ -O3 -std=c++11 -DEN_TEST_CPU -I$(AMDAPP_ROOT)/include $< -L$(AMDAPP_ROOT)/lib/x86_64 -lOpenCL -o $@

gpu_ocl.out: force_ocl.cpp
	g++ -O3 -std=c++11 -I$(AMDAPP_ROOT)/include $< -L$(AMDAPP_ROOT)/lib/x86_64 -lOpenCL -o $@

kernel.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

kernel_aar.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) -ptx $< $(LIBRARY) -o $@

clean:
	rm -f $(TARGET) $(CUDA) $(PTX) $(OCL)

test: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out
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

cuda_bench: aos_intrin.out soa_intrin.out aos_pair.out soa_pair.out gpu_cuda_aar.out gpu_cuda.out
	./aos_intrin.out
	./soa_intrin.out
	./aos_pair.out
	./soa_pair.out
	./gpu_cuda_aar.out
	./gpu_cuda.out

clear:
	rm -f $(CACHE_FILE)
