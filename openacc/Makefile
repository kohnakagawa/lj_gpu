# NOTE: pgcpp is deprecated.
CXX = pgc++

TARGET = gpu_oacc_soa_ref.out gpu_oacc_soa_trans.out gpu_oacc_aos_ref.out \
	gpu_oacc_aos_warp_unroll.out gpu_oacc_aos_trans.out gpu_oacc_aos_trans_tuned.out

OPT_GPU = -ta=nvidia,cc35,keep,cuda7.5,ptxinfo
# OPT_GPU = -ta=nvidia,cc60,keep,cuda8.0,ptxinfo

OACC_FLAGS = -acc  -Minfo=accel $(OPT_GPU)
PGI_FLAGS  = -fast -Minform=warn
CXX_FLAGS = -std=c++11 $(PGI_FLAGS) $(OACC_FLAGS)

all: $(TARGET)

gpu_oacc_soa_ref.out: force_oacc_soa.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_REF $< -o $@

gpu_oacc_soa_trans.out: force_oacc_soa.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_TRANS $< -o $@

gpu_oacc_aos_ref.out: force_oacc_aos.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_REF $< -o $@

gpu_oacc_aos_warp_unroll.out: force_oacc_aos.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_WARP $< -o $@

gpu_oacc_aos_trans.out: force_oacc_aos.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_TRANS $< -o $@

gpu_oacc_aos_trans_tuned.out: force_oacc_aos.cpp
	$(CXX) $(CXX_FLAGS) -DOACC_TRANS_TUNED $< -o $@

clean:
	rm -f *~ $(TARGET)
