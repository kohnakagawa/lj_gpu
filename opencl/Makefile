CACHE_FILE = .cache_pair_half.dat .cache_pair_all.dat

OCL = gpu_ocl.out

WARNINGS = -Wall -Wextra
AMDAPP_ROOT=/opt/AMDAPPSDK-3.0

all: $(OCL)

gpu_ocl.out: force_ocl.cpp
	g++ $(WARNINGS) -O3 -std=c++11 -isystem $(AMDAPP_ROOT)/include $< -L$(AMDAPP_ROOT)/lib/x86_64 -lOpenCL -o $@

clean:
	rm -f $(OCL) *~ *.core

clear:
	rm -f $(CACHE_FILE)
