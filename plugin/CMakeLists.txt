# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PoolingPlugin LANGUAGES CXX CUDA)

# Configurable variable message macro
macro(var_found_message var)
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    var_found_message(${var})
endmacro()

# Protocol Buffers
find_package(Protobuf REQUIRED)
if(Protobuf_FOUND)
    var_found_message(Protobuf_VERSION)
    var_found_message(Protobuf_INCLUDE_DIRS)
    var_found_message(Protobuf_LIBRARIES)
endif()

# CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    var_found_message(CUDA_VERSION)
    var_found_message(CUDA_INCLUDE_DIRS)
endif()

# CuDNN
include(./FindCuDNN.cmake)
if(CUDNN_FOUND)
    var_found_message(CUDNN_VERSION)
    var_found_message(CUDNN_INCLUDE_DIRS)
    var_found_message(CUDNN_LIBRARIES)
    var_found_message(CUDNN_LIBRARY_DIRS)
endif()

# TensorRT
include(./FindTensorRT.cmake)
if(TensorRT_FOUND)
    var_found_message(TensorRT_VERSION_STRING)
    var_found_message(TensorRT_INCLUDE_DIRS)
    var_found_message(TensorRT_LIBRARIES)
endif()

# -------- CONFIGURATION --------

# -------- BUILDING --------

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

# Add include directories
include_directories(
    ${Protobuf_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/Common/
)

# Define clip plugin library target
add_library(PoolingPlugin MODULE
    ${CMAKE_SOURCE_DIR}/PoolingAlgo.cu
    ${CMAKE_SOURCE_DIR}/CudaPooling.cpp
    ${CMAKE_SOURCE_DIR}/CudaPooling.h
    ${CMAKE_SOURCE_DIR}/trt_plugin.pb.cpp
    ${CMAKE_SOURCE_DIR}/PoolingPlugin.cpp
    ${CMAKE_SOURCE_DIR}/Pooling.h
    ${CMAKE_SOURCE_DIR}/CuDnnPooling.cpp
    ${CMAKE_SOURCE_DIR}/CuDnnPooling.h
    ${CMAKE_SOURCE_DIR}/CopyPlugin.cpp
)

# Use C++11
target_compile_features(PoolingPlugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(PoolingPlugin PRIVATE
    ${Protobuf_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${TensorRT_LIBRARIES}
)

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(PoolingPlugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
