TARGET= aos.out aos_pair.out aos_intrin.out soa.out soa_pair.out soa_intrin.out gpu.out cpu_ref.out kernel.ptx

WARNINGS = -Wextra -Wunused-variable -Wsign-compare -Wnon-virtual-dtor -Woverloaded-virtual
OPT_FLAGS = -O3 -funroll-loops -ffast-math

gpu_profile = yes

CUDA_HOME=/usr/local/cuda
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= -O3 -std=c++11 -arch=sm_35 -Xcompiler "-std=c++11 $(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
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

cpu_ref.out: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) -DEN_TEST $(INCLUDE) $< $(LIBRARY) -o $@

kernel.ptx: force_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

clean:
	rm -f $(TARGET)

test: aos_pair.out aos_intrin.out soa_pair.out soa_intrin.out
	./aos_pair.out > aos_pair.dat
	./aos_intrin.out > aos_intrin.dat
	diff aos_pair.dat aos_intrin.dat
	./soa_pair.out > soa_pair.dat
	./soa_intrin.out > soa_intrin.dat
	diff soa_pair.dat soa_intrin.dat

test_gpu: cpu_ref.out gpu.out
	./cpu_ref.out > cpu_ref.txt
	./gpu.out > gpu_org.txt
	diff cpu_ref.txt gpu_org.txt
