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

  #define DEV_MALLOC        cudaMalloc
  #define DEV_FREE          cudaFree
  #define DEV_MEMCPY        cudaMemcpy
  #define DEV_MEMSET        cudaMemset
  #define DEV_DEVICE_SYNC   cudaDeviceSynchronize
  #define RUNTIME_GET_LAST_ERR  cudaGetLastError


  #define MEMCPY_H2D        cudaMemcpyHostToDevice
  #define MEMCPY_D2H        cudaMemcpyDeviceToHost
  #define MEMCPY_D2D        cudaMemcpyDeviceToDevice
  
#if defined(PLATFORM_NVIDIA)
extern "C" cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}
#endif

// kernel launch check：先抓 launch error，再 sync（便于区分 launch vs runtime）
#define KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                \
    auto e = RUNTIME_GET_LAST_ERR();                                      \
    if (e != RUNTIME_SUCCESS_CODE) {                                           \
      std::cerr << "Kernel launch error: " << RUNTIME_GET_ERROR_STR(e) << "\n"; \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
    RUNTIME_CHECK(DEV_DEVICE_SYNC());                                 \
  } while (0)

#elif defined(PLATFORM_MOORE)
#include <musa_runtime.h>
#define RUNTIME_ERR_TYPE musaError_t
#define RUNTIME_SUCCESS_CODE musaSuccess
#define RUNTIME_GET_ERROR_STR musaGetErrorString

#elif defined(PLATFORM_METAX)
#include <mcr/mc_runtime.h>
#define RUNTIME_ERR_TYPE mcError_t
#define RUNTIME_SUCCESS_CODE mcSuccess
#define RUNTIME_GET_ERROR_STR mcGetErrorString

#else
#error "Unknown PLATFORM for RUNTIME_CHECK"
#endif

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T>
__device__ __forceinline__ float to_float_dev(T x) {
  if constexpr (std::is_same_v<T, half>) return __half2float(x);
  else return (float)x;
}
