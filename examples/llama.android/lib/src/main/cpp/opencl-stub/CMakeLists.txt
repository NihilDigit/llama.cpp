cmake_minimum_required(VERSION 3.10)
project(opencl_stub C)

# Create a minimal stub library that provides OpenCL symbols for LINKING ONLY
# The real implementation will be loaded from the device at runtime

add_library(OpenCL SHARED opencl_stub.c)
target_include_directories(OpenCL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../opencl-headers)
