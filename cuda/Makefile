CACHE_FILE = .cache_pair_half.dat .cache_pair_all.dat

PTX = kernel.ptx kernel_aar.ptx
SASS = kernel.sass
CUBIN = kernel.cubin
CUDA = gpu_cuda.out gpu_cuda_aar.out gpu_cuda_test.out

WARNINGS = -Wall -Wextra
GCC_FLAGS = -O3 -march=native

# CUDA_HOME=/usr/local/cuda
CUDA_HOME=$(CUDA_PATH)

NVCC=$(CUDA_HOME)/bin/nvcc

# ARCH=-arch=sm_60
ARCH=-arch=sm_35

NVCCFLAGS= -O3 -std=c++11 $(ARCH) -Xcompiler "$(WARNINGS) $(GCC_FLAGS)" -ccbin=g++ -Xptxas -v
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc

all: $(CUDA)
ptx: $(PTX)
sass: $(CUBIN) $(SASS)

gpu_cuda.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

gpu_cuda_aar.out: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) $< $(LIBRARY) -o $@

kernel.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

kernel_aar.ptx: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) -DEN_ACTION_REACTION $(INCLUDE) -ptx $< $(LIBRARY) -o $@

kernel.cubin: force_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -cubin $< $(LIBRARY) -o $@

kernel.sass: kernel.cubin
	$(CUDA_HOME)/bin/cuobjdump -sass $< > $@

clean:
	rm -f $(AVX) $(CUDA) $(PTX) $(OCL) $(OACC) *.gpu *.ptx *.sass *.cubin *~ *.core

clear:
	rm -f $(CACHE_FILE)
