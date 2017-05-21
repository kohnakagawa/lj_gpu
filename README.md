# SIMTization for Force Calculation of Lennard-Jones Potential
### This project is forked from https://github.com/kaityo256/lj_simd

## Usage 
### NVIDIAのGPUで実行する場合 (CUDA)
    $ make cuda
    $ make test_cuda

### NVIDIAのGPUで実行する場合 (OpenACC)
    $ make oacc
    $ make test_oacc
    
### AMDのGPUで実行する場合 (OpenCL)
    $ make ocl
    $ make test_ocl
    
### IntelのCPUで実行する場合
    $ make avx
    $ make test_avx
