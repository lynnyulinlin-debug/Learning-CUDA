#pragma once

#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>

#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define RUNTIME_ERR_TYPE cudaError_t
#define RUNTIME_SUCCESS_CODE cudaSuccess
#define RUNTIME_GET_ERROR_STR cudaGetErrorString
#define RUNTIME_GET_LAST_ERR  cudaGetLastError

#define RUNTIME_MALLOC        cudaMalloc
#define RUNTIME_FREE          cudaFree
#define RUNTIME_MEMCPY        cudaMemcpy
#define RUNTIME_MEMSET        cudaMemset

#define RUNTIME_DEVICE_SYNC   cudaDeviceSynchronize

#define MEMCPY_H2D        cudaMemcpyHostToDevice
#define MEMCPY_D2H        cudaMemcpyDeviceToHost
//#define MEMCPY_D2D        cudaMemcpyDeviceToDevice
//#define MEMCPY_H2H        cudaMemcpyHostToHost
  
#if defined(PLATFORM_NVIDIA)
extern "C" RUNTIME_ERR_TYPE cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}
#endif

#elif defined(PLATFORM_MOORE)
#include <musa_runtime.h>
#include <musa_fp16.h>
#define RUNTIME_ERR_TYPE musaError_t
#define RUNTIME_SUCCESS_CODE musaSuccess
#define RUNTIME_GET_ERROR_STR musaGetErrorString
#define RUNTIME_GET_LAST_ERR  mudaGetLastError

#define RUNTIME_MALLOC        mudaMalloc
#define RUNTIME_FREE          mudaFree
#define RUNTIME_MEMCPY        mudaMemcpy
#define RUNTIME_MEMSET        mudaMemset

#define RUNTIME_DEVICE_SYNC   mudaDeviceSynchronize

#define MEMCPY_H2D        mudaMemcpyHostToDevice
#define MEMCPY_D2H        mudaMemcpyDeviceToHost
//#define MEMCPY_D2D        mudaMemcpyDeviceToDevice
//#define MEMCPY_H2H        mudaMemcpyHostToHost

#elif defined(PLATFORM_METAX)
#include <mcr/mc_runtime.h>
#include <mcr/mc_fp16.h>
#define RUNTIME_ERR_TYPE mcError_t
#define RUNTIME_SUCCESS_CODE mcSuccess
#define RUNTIME_GET_ERROR_STR mcGetErrorString
#define RUNTIME_GET_LAST_ERR  mcdaGetLastError

#define RUNTIME_MALLOC        mcdaMalloc
#define RUNTIME_FREE          mcdaFree
#define RUNTIME_MEMCPY        mcdaMemcpy
#define RUNTIME_MEMSET        mcdaMemset

#define RUNTIME_DEVICE_SYNC   mcdaDeviceSynchronize

#define MEMCPY_H2D        mcdaMemcpyHostToDevice
#define MEMCPY_D2H        mcdaMemcpyDeviceToHost
//#define MEMCPY_D2D        mcdaMemcpyDeviceToDevice
//#define MEMCPY_H2H        mcdaMemcpyHostToHost


#else
#error "Unknown PLATFORM for RUNTIME_CHECK"
#endif

#define KERNEL_LAUNCH_CHECK()                                                 \
  do {                                                                        \
    RUNTIME_ERR_TYPE err = RUNTIME_GET_LAST_ERR();                            \
    if (err != RUNTIME_SUCCESS_CODE) {                                        \
      std::cerr << "Kernel launch error at " << __FILE__ << ":" << __LINE__   \
                << " - " << RUNTIME_GET_ERROR_STR(err) << "\n";               \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
    RUNTIME_CHECK(RUNTIME_DEVICE_SYNC());                                     \
  } while (0)

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// device helpers
// 注意：flashAttention 为了数值一致性，统一先转 float 再算，再转回。
// 注意：C++11 不支持 if constexpr，使用模板特化替代
// ============================================================================

#if __cplusplus >= 201703L
template <typename T>
__device__ __forceinline__ float to_float_dev(T x) {
  if constexpr (std::is_same_v<T, half>) return __half2float(x);
  else return (float)x;
}

template <typename T>
__device__ __forceinline__ T from_float_dev(float x) {
  if constexpr (std::is_same_v<T, half>) return __float2half(x);
  else return (T)x;
}
#else
  // C++11/C++14 版本：使用模板特化，兼容旧标准
  template<typename T> 
__device__ __forceinline__ float to_float_dev(T x) {
  return static_cast<float>(x);
}
template<> 
__device__ __forceinline__ float to_float_dev<half>(half x) {
  return __half2float(x);
}

template<typename T> 
__device__ __forceinline__ T from_float_dev(float x) {
  return static_cast<T>(x);
}
template<> 
__device__ __forceinline__ half from_float_dev<half>(float x) {
  return __float2half(x);
}

#endif

