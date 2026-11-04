// а1
#pragma once
#include <cstdint>
#if !defined(TIER_CUDA)

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#if __has_include(<hipblas/hipblas.h>)          // а2
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif


// а3
#if !defined(TIER_NO_LT) && __has_include(<hipblaslt/hipblaslt.h>)
#include <hipblaslt/hipblaslt.h>
#define TIER_LT 1
#else
#define TIER_LT 0
#endif
#define TIER_GPU "AMD (HIP)"

#else   // а4

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdint>
#define TIER_GPU "nvidia (CUDA)"
#define TIER_LT 1

// а5
using hipError_t        = cudaError_t;
using hipStream_t       = cudaStream_t;
using hipEvent_t        = cudaEvent_t;
using hipDeviceProp_t   = cudaDeviceProp;
using hipblasHandle_t   = cublasHandle_t;
using hipblasStatus_t   = cublasStatus_t;
using hipblasOperation_t= cublasOperation_t;
using hipDataType       = cudaDataType_t;
using hipblasLtHandle_t              = cublasLtHandle_t;
using hipblasLtMatmulDesc_t          = cublasLtMatmulDesc_t;
using hipblasLtMatrixLayout_t        = cublasLtMatrixLayout_t;
using hipblasLtMatmulPreference_t    = cublasLtMatmulPreference_t;
using hipblasLtMatmulAlgo_t          = cublasLtMatmulAlgo_t;
using hipblasLtMatmulHeuristicResult_t = cublasLtMatmulHeuristicResult_t;

#define hipSuccess              cudaSuccess
#define hipErrorOutOfMemory     cudaErrorMemoryAllocation
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipStreamNonBlocking    cudaStreamNonBlocking
#define hipEventDisableTiming   cudaEventDisableTiming
#define HIPBLAS_OP_N            CUBLAS_OP_N
#define HIPBLAS_OP_T            CUBLAS_OP_T
#define HIPBLAS_STATUS_SUCCESS  CUBLAS_STATUS_SUCCESS
#define HIPBLAS_COMPUTE_32F     CUBLAS_COMPUTE_32F
#define HIP_R_32F               CUDA_R_32F
#define HIP_R_16BF              CUDA_R_16BF
#define HIP_R_16F               CUDA_R_16F
#define HIPBLASLT_MATMUL_DESC_TRANSA CUBLASLT_MATMUL_DESC_TRANSA
#define HIPBLASLT_MATMUL_DESC_TRANSB CUBLASLT_MATMUL_DESC_TRANSB
#define HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES

// а6
#define hipMalloc            cudaMalloc
#define hipFree              cudaFree
#define hipMemcpy            cudaMemcpy
#define hipMemcpyAsync       cudaMemcpyAsync
#define hipMemcpy2D          cudaMemcpy2D
#define hipMemset            cudaMemset
#define hipMemGetInfo        cudaMemGetInfo
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamCreate      cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamWaitEvent   cudaStreamWaitEvent
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventRecord       cudaEventRecord
#define hipGetLastError      cudaGetLastError
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipFuncGetAttributes cudaFuncGetAttributes
#define hipFuncAttributes    cudaFuncAttributes
#define hipMemcpyToSymbol    cudaMemcpyToSymbol
#define hipMemcpyFromSymbol  cudaMemcpyFromSymbol
#define hipStreamDestroy     cudaStreamDestroy
#define hipEventDestroy      cudaEventDestroy
#define hipEventSynchronize  cudaEventSynchronize
#define hipEventElapsedTime  cudaEventElapsedTime
#define hipEventCreate       cudaEventCreate
#define hipGetErrorString    cudaGetErrorString
#define hipSetDevice         cudaSetDevice
#define hipGetDeviceCount    cudaGetDeviceCount
#define hipDeviceReset       cudaDeviceReset
#define hipMemcpyKind        cudaMemcpyKind
#define hipHostRegister      cudaHostRegister
#define hipHostUnregister    cudaHostUnregister
#define HIP_SYMBOL(x)        (x)
static inline cudaError_t hipHostMalloc(void** p, size_t n){ return cudaHostAlloc(p,n,cudaHostAllocDefault); }
template<class T> static inline cudaError_t hipHostMalloc(T** p, size_t n){ return cudaHostAlloc((void**)p,n,cudaHostAllocDefault); }
#define hipHostFree          cudaFreeHost

// в1
#define hipblasCreate        cublasCreate
#define hipblasDestroy       cublasDestroy
#define hipblasSetStream     cublasSetStream
#define hipblasSgemm         cublasSgemm
#define hipblasDgemm         cublasDgemm
#define hipblasLtDestroy     cublasLtDestroy
#define hipblasLtMatmulPreferenceAttributes_t cublasLtMatmulPreferenceAttributes_t

// в2
#define hipblasLtCreate                   cublasLtCreate
#define hipblasLtMatmul                   cublasLtMatmul
#define hipblasLtMatmulDescCreate         cublasLtMatmulDescCreate
#define hipblasLtMatmulDescDestroy        cublasLtMatmulDescDestroy
#define hipblasLtMatmulDescSetAttribute   cublasLtMatmulDescSetAttribute
#define hipblasLtMatrixLayoutCreate       cublasLtMatrixLayoutCreate
#define hipblasLtMatrixLayoutDestroy      cublasLtMatrixLayoutDestroy
#define hipblasLtMatmulPreferenceCreate   cublasLtMatmulPreferenceCreate
#define hipblasLtMatmulPreferenceDestroy  cublasLtMatmulPreferenceDestroy
#define hipblasLtMatmulPreferenceSetAttribute cublasLtMatmulPreferenceSetAttribute
#define hipblasLtMatmulAlgoGetHeuristic   cublasLtMatmulAlgoGetHeuristic

// г1
template<class T> __device__ inline T __shfl_down(T v, unsigned d, int w=32){ return __shfl_down_sync(0xffffffffu, v, d, w); }
template<class T> __device__ inline T __shfl_xor(T v, int m, int w=32){ return __shfl_xor_sync(0xffffffffu, v, m, w); }
template<class T> __device__ inline T __shfl(T v, int src, int w=32){ return __shfl_sync(0xffffffffu, v, src, w); }

// ж1
struct hip_bfloat16 {
    __nv_bfloat16 v;
    __host__ __device__ hip_bfloat16() {}
    __host__ __device__ hip_bfloat16(float f) : v(__float2bfloat16(f)) {}
    __host__ __device__ operator float() const { return __bfloat162float(v); }
};
static_assert(sizeof(hip_bfloat16)==2, "BF16 должен занимать 2 байта");

#endif

// з1
struct tier_fp16 {
    uint16_t bits;
    __host__ __device__ tier_fp16() : bits(0) {}
    __host__ __device__ tier_fp16(float f) : bits(from_float(f)) {}
    __host__ __device__ operator float() const { return to_float(bits); }

    __host__ __device__ static uint16_t from_float(float f) {
        union { float f; uint32_t u; } v; v.f=f;
        const uint32_t sign=(v.u>>16)&0x8000u;
        const uint32_t exp=(v.u>>23)&0xffu;
        uint32_t mant=v.u&0x7fffffu;
        if(exp==0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
        int e=(int)exp-127+15;
        if(e>=31) return (uint16_t)(sign|0x7c00u);
        if(e<=0) {
            if(e < -10) return (uint16_t)sign;
            mant |= 0x800000u;
            const int shift=14-e;
            uint32_t q=mant>>shift;
            uint32_t rem=mant&((1u<<shift)-1u), halfway=1u<<(shift-1);
            if(rem>halfway || (rem==halfway && (q&1u))) ++q;
            return (uint16_t)(sign|q);
        }
        uint32_t q=mant>>13, rem=mant&0x1fffu;
        if(rem>0x1000u || (rem==0x1000u && (q&1u))) {
            ++q;
            if(q==0x400u) { q=0; ++e; if(e>=31) return (uint16_t)(sign|0x7c00u); }
        }
        return (uint16_t)(sign|((uint32_t)e<<10)|q);
    }
    __host__ __device__ static float to_float(uint16_t h) {
        const uint32_t sign=((uint32_t)h&0x8000u)<<16;
        uint32_t exp=((uint32_t)h>>10)&0x1fu, mant=(uint32_t)h&0x3ffu;
        union { uint32_t u; float f; } v;
        if(exp==0) {
            if(!mant) { v.u=sign; return v.f; }
            int e=-14;
            while((mant&0x400u)==0) { mant<<=1; --e; }
            mant&=0x3ffu;
            v.u=sign|((uint32_t)(e+127)<<23)|(mant<<13);
            return v.f;
        }
        if(exp==31) { v.u=sign|0x7f800000u|(mant<<13); return v.f; }
        v.u=sign|((exp+112u)<<23)|(mant<<13);
        return v.f;
    }
};
static_assert(sizeof(tier_fp16)==2, "FP16 storage must occupy two bytes");

#if !defined(TIER_CUDA) && !TIER_LT
// з2
#include <cstdint>
typedef void* hipblasLtHandle_t; typedef void* hipblasLtMatmulDesc_t; typedef void* hipblasLtMatrixLayout_t; typedef void* hipblasLtMatmulPreference_t;
struct hipblasLtMatmulAlgo_t { uint64_t data[8]; size_t max_workspace_bytes; };
struct hipblasLtMatmulHeuristicResult_t { hipblasLtMatmulAlgo_t algo; size_t workspaceSize; hipblasStatus_t state; float wavesCount; int reserved[4]; };
enum { HIPBLASLT_MATMUL_DESC_TRANSA = 0, HIPBLASLT_MATMUL_DESC_TRANSB = 1, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1 };
#ifndef HIPBLAS_COMPUTE_32F
#define HIPBLAS_COMPUTE_32F 0
#endif
#define TIER_LT_NO return HIPBLAS_STATUS_NOT_SUPPORTED
static inline hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* h){ *h=nullptr; TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* d, int, int){ *d=nullptr; TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t, int, const void*, size_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* l, int, uint64_t, uint64_t, int64_t){ *l=nullptr; TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* p){ *p=nullptr; TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t, int, const void*, size_t){ TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t*, int* n){ *n=0; TIER_LT_NO; }
static inline hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*, const void*, hipblasLtMatrixLayout_t, const void*,
        hipblasLtMatrixLayout_t, const void*, const void*, hipblasLtMatrixLayout_t, void*, hipblasLtMatrixLayout_t, const hipblasLtMatmulAlgo_t*,
        void*, size_t, hipStream_t){ TIER_LT_NO; }
#endif

// ч1
#if defined(TIER_CUDA)
#define TIER_TRAP() __trap()
#else
#define TIER_TRAP() __builtin_trap()
#endif

// ч2
#ifndef TIER_WARP
#define TIER_WARP 32
#endif
