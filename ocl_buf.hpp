#pragma once

#include "ocl_util.hpp"

template <typename T>
struct ocl_buf {
  int size = -1;
  cl::Buffer dev_buf;
  T* host_buf = nullptr;

  ocl_buf() {}
  ~ocl_buf() { Deallocate(); }

  // disable copy constructor
  const ocl_buf<T>& operator = (const ocl_buf<T>& obj) = delete;
  ocl_buf<T>(const ocl_buf<T>& obj) = delete;

  // disable move constructor
  ocl_buf<T>& operator = (ocl_buf<T>&& obj) = delete;
  ocl_buf<T>(ocl_buf<T>&& obj) = delete;

  void Allocate(const std::size_t size_,
                cl::Context& context,
                const cl_mem_flags flag) {
    size = size_;
    host_buf = new T [size];
    cl_int status = 0;
    dev_buf  = cl::Buffer(context, flag, size * sizeof(T), nullptr, &status);
    checkErr(status, "ocl_buf::Allocate");
  }

  void Host2dev(const cl::CommandQueue& queue,
                const cl_bool blocking_write) {
    queue.enqueueWriteBuffer(dev_buf, blocking_write, 0, size * sizeof(T), host_buf);
  }

  void Dev2host(const cl::CommandQueue& queue,
                const cl_bool blocking_read) {
    queue.enqueueReadBuffer(dev_buf, blocking_read, 0, size * sizeof(T), host_buf);
  }

  const T& operator [] (const int i) const {
    return host_buf[i];
  }

  T& operator [] (const int i) {
    return host_buf[i];
  }

  cl::Buffer& GetDevBuffer() {
    return dev_buf;
  }

private:
  void Deallocate() {
    delete [] host_buf;
  }
};
