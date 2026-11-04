#pragma once
// а1
#include <cstddef>
#include <cstdint>
#include <cmath>
#define __global__   __attribute__((global))
#define __device__   __attribute__((device))
#define __host__     __attribute__((host))
#define __shared__   __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
struct __cuda_dim { unsigned x,y,z; __cuda_dim()=default; __cuda_dim(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
typedef __cuda_dim dim3;
__device__ extern const __cuda_dim blockIdx, blockDim, threadIdx, gridDim;
__device__ extern const int warpSize;
__device__ void __syncthreads();
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t=0, void* =nullptr);
#include <cstdio>
extern "C" __device__ int printf(const char*, ...);
extern "C" unsigned cudaConfigureCall(dim3, dim3, size_t=0, void* =nullptr);
__device__ float atomicAdd(float*, float);
__device__ int atomicAdd(int*, int);
__device__ unsigned atomicAdd(unsigned*, unsigned);
__device__ unsigned long long atomicAdd(unsigned long long*, unsigned long long);
__device__ double atomicAdd(double*, double);
template<class T> __device__ T __shfl_down_sync(unsigned, T, unsigned, int=32);
template<class T> __device__ T __shfl_xor_sync(unsigned, T, int, int=32);
template<class T> __device__ T __shfl_sync(unsigned, T, int, int=32);
typedef enum { cudaSuccess=0, cudaErrorMemoryAllocation=2 } cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st*  cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum { cudaStreamNonBlocking=1, cudaEventDisableTiming=2, cudaHostAllocDefault=0 };
typedef enum { CUDA_R_32F=0, CUDA_R_16BF=1, CUDA_R_16F=2 } cudaDataType_t;
struct cudaDeviceProp { char name[256]; int multiProcessorCount, warpSize, major, minor, integrated, pciDomainID, pciBusID, pciDeviceID; size_t totalGlobalMem, sharedMemPerBlock; };
cudaError_t cudaMalloc(void**, size_t);
template<class T> cudaError_t cudaMalloc(T** p, size_t n){ return cudaMalloc((void**)p,n); }
cudaError_t cudaFree(void*);
cudaError_t cudaHostAlloc(void**, size_t, unsigned);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t=nullptr);
cudaError_t cudaMemcpy2D(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaMemGetInfo(size_t*, size_t*);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaSetDevice(int);
__device__ void __trap();
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned=0);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t=nullptr);
const char* cudaGetErrorString(cudaError_t);
template<class T> cudaError_t cudaMemcpyToSymbol(const T&, const void*, size_t, size_t=0, cudaMemcpyKind=cudaMemcpyHostToDevice);
typedef unsigned short ushort;
struct ushort4 { unsigned short x,y,z,w; };
struct float2 { float x,y; }; struct float4 { float x,y,z,w; };
__device__ float4 make_float4(float,float,float,float);
__device__ float __uint_as_float(unsigned);
__device__ float rsqrtf(float); __device__ float powf(float,float); __device__ float cosf(float);
__device__ float sinf(float); __device__ float expf(float); __device__ float logf(float);
__device__ float sqrtf(float); __device__ float fabsf(float); __device__ float tanhf(float);
__device__ float fmaf(float,float,float); __device__ float fmax(float,float); __device__ float fmin(float,float);
__device__ float fmaxf(float,float); __device__ float fminf(float,float);
__device__ double exp(double); __device__ double log(double); __device__ double sqrt(double); __device__ double fabs(double);
template<class T> __device__ T max(T a,T b){ return a>b?a:b; }
template<class T> __device__ T min(T a,T b){ return a<b?a:b; }
