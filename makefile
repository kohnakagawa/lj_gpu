TARGET= aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out gpu.out gpu_test.out gpu_aar.out cpu_ref.out cpu_aar.out kernel.ptx kernel_aar.ptx
CACHE_FILE = .cache_pair_half.dat .cache_pair_all.dat

WARNINGS = -Wall -Wextra -Wunused-variable -Wsign-compare
OPT_FLAGS = -O3 -funroll-loops -ffast-math

gpu_profile = yes

CUDA_HOME=/usr/local/cuda
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= -O3 -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc
ifeq ($(gpu_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

all: $(TARGET)

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

gpu.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

gpu_test.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_GPU $(INCLUDE) $< $(LIBRARY) -o $@

gpu_aar.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) $< $(LIBRARY) -o $@

cpu_ref.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_CPU $(INCLUDE) $< $(LIBRARY) -o $@

cpu_aar.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST_CPU -DEN_ACTION_REACTION $(INCLUDE) $< $(LIBRARY) -o $@

kernel.ptx: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

kernel_aar.ptx: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) -ptx $< $(LIBRARY) -o $@

clean:
	rm -f $(TARGET)

test: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out
	./aos_pair.out > aos_pair.dat
	./aos_intrin.out > aos_intrin.dat
	diff aos_pair.dat aos_intrin.dat
	./soa_pair.out > soa_pair.dat
	./soa_intrin.out > soa_intrin.dat
	diff soa_pair.dat soa_intrin.dat

test_gpu: cpu_aar.out cpu_ref.out gpu_test.out gpu_aar.out
	./cpu_aar.out > cpu_aar.txt
	./cpu_ref.out > cpu_ref.txt
	./gpu_test.out > gpu_test.txt
	./gpu_aar.out > gpu_aar.txt
	diff cpu_aar.txt cpu_ref.txt
	diff cpu_ref.txt gpu_test.txt
	diff gpu_test.txt gpu_aar.txt

bench: gpu.out gpu_aar.out
	./gpu_aar.out
	./gpu.out

clear:
	rm -f $(CACHE_FILE)
