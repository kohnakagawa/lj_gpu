#pragma once

#include <string>
#include <map>
#include "ocl_util.hpp"

class OclDevice {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program::Sources sources;
  cl::Program program;
  std::map<std::string, cl::Kernel> kernels;
  std::vector<std::string> kernel_names;

  void GetPlatformAll() {
    checkErr(cl::Platform::get(&platforms), "cl::Platform::get");
    checkErr(platforms.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
  }

  void DisplayPlatFormInfo(const cl::Platform& plt,
                           const cl_platform_info id,
                           const std::string str) {
    std::string info;
    checkErr(plt.getInfo(id, &info), "cl::Platform::getInfo" );
    std::cerr << str << ": " << info << "\n";
  }

  void DisplayPlatFormInfoAll() {
    std::cerr << "Platform number is: " << platforms.size() << "\n";
    for (auto& plt : platforms) {
      DisplayPlatFormInfo(plt, CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
      DisplayPlatFormInfo(plt, CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
      DisplayPlatFormInfo(plt, CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
      DisplayPlatFormInfo(plt, CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
      DisplayPlatFormInfo(plt, CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
    }
  }

  void SetPlatform() {
    platform = *(platforms.rbegin());
  }

  void GetDeviceAll(const cl_device_type type) {
    checkErr(platform.getDevices(type, &devices), "cl::Platform::getDevices");
  }

  void DisplayDeviceInfo(const cl::Device& dev,
                         const cl_device_info name,
                         const std::string str) {
    std::string info;
    checkErr(dev.getInfo(name, &info), "cl::Device::getInfo");
    std::cerr << str << ": " << info << "\n";
  }

  void DisplayDeviceInfoAll() {
    std::cerr << "Number of devices " << devices.size() << " is found.\n";
    for (auto& dev : devices) {
      DisplayDeviceInfo(dev, CL_DEVICE_NAME,   "Device name");
      DisplayDeviceInfo(dev, CL_DEVICE_VENDOR, "Device vendor");
      DisplayDeviceInfo(dev, CL_DEVICE_PROFILE, "Device profile");
      DisplayDeviceInfo(dev, CL_DEVICE_VERSION, "Device version");
      DisplayDeviceInfo(dev, CL_DRIVER_VERSION, "Driver version");
      DisplayDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, "OpenCL C version");
      DisplayDeviceInfo(dev, CL_DEVICE_EXTENSIONS, "OpenCL device extensions");
    }
  }

  void SetDevice() {
    device = *(devices.rbegin());
  }

public:
  OclDevice() {}
  ~OclDevice() {}

  void Initialize() {
    GetPlatformAll();
    DisplayPlatFormInfoAll();
    SetPlatform();
    GetDeviceAll(CL_DEVICE_TYPE_GPU);
    DisplayDeviceInfoAll();
    SetDevice();
  }

  void AddProgramSource(const char* source_str) {
    sources.push_back(std::make_pair(source_str, std::strlen(source_str)));
  }

  void AddKernelName(const std::string name) {
    kernel_names.push_back(name);
  }

  void BuildProgram() {
    program = cl::Program(context, sources);
    try {
      program.build(devices);
    } catch (cl::Error err) {
      if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
      } else {
        std::cerr << "Unknown error #" << err.err() << " " << err.what() << "\n";
      }
    }
  }

  void CreateContext() {
    context = cl::Context(device);
  }

  void CreateKernels() {
    for (const auto& name : kernel_names) {
      kernels[name] = cl::Kernel(program, name.c_str());
    }
  }

  cl::Kernel& GetKernel(const std::string name) {
    return kernels.at(name);
  }

  template <typename T>
  void SetFunctionArg(const std::string name,
                      const int pos,
                      T& buf) {
    kernels.at(name).setArg(pos, buf);
  }

  cl::Device& GetCurrentDevice() {
    return device;
  }

  cl::Context& GetCurrentContext() {
    return context;
  }
};
