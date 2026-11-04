#pragma once
#include <cuda_runtime.h>
struct __nv_bfloat16 { unsigned short d; __host__ __device__ __nv_bfloat16(){} };
__host__ __device__ __nv_bfloat16 __float2bfloat16(float);
__host__ __device__ float __bfloat162float(__nv_bfloat16);
