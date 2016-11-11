#pragma once

#include <iostream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

static inline void checkErr(cl_int err, const char* name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

