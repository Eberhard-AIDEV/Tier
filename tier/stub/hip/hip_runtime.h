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
struct __hip_dim { unsigned x, y, z; __hip_dim()=default; __hip_dim(unsigned a, unsigned b=1, unsigned c=1):x(a),y(b),z(c){} };
typedef __hip_dim dim3;
__device__ extern const __hip_dim blockIdx, blockDim, threadIdx, gridDim;
__device__ void __syncthreads();
__device__ float atomicAdd(float*, float);
__device__ int   atomicAdd(int*, int);
__device__ unsigned atomicAdd(unsigned*, unsigned);
__device__ float __shfl_down(float, unsigned, int=64);
// б1
extern "C" int __hipPushCallConfiguration(__hip_dim, __hip_dim, size_t=0, void* =nullptr);
typedef enum { hipSuccess=0, hipErrorOutOfMemory=2 } hipError_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t*  hipEvent_t;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, hipMemcpyDefault };
enum { hipStreamNonBlocking=1, hipEventDisableTiming=2, hipHostMallocDefault=0 };
struct hipDeviceProp_t { char name[256]; int multiProcessorCount, warpSize, major, minor, integrated, pciDomainID, pciBusID, pciDeviceID; size_t totalGlobalMem, sharedMemPerBlock; char gcnArchName[256]; };
hipError_t hipMalloc(void**, size_t);
template<class T> hipError_t hipMalloc(T** p, size_t n){ return hipMalloc((void**)p, n); }
hipError_t hipFree(void*);
hipError_t hipHostMalloc(void**, size_t, unsigned=0);
template<class T> hipError_t hipHostMalloc(T** p, size_t n, unsigned f=0){ return hipHostMalloc((void**)p, n, f); }
hipError_t hipHostFree(void*);
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t=nullptr);
hipError_t hipMemcpy2D(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind);
hipError_t hipMemset(void*, int, size_t);
hipError_t hipMemGetInfo(size_t*, size_t*);
hipError_t hipDeviceSynchronize();
hipError_t hipGetLastError();
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipGetDeviceCount(int*);
hipError_t hipSetDevice(int);
hipError_t hipStreamCreate(hipStream_t*);
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned=0);
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned);
hipError_t hipEventRecord(hipEvent_t, hipStream_t=nullptr);
const char* hipGetErrorString(hipError_t);
template<class T> hipError_t hipMemcpyToSymbol(const T&, const void*, size_t, size_t=0, hipMemcpyKind=hipMemcpyHostToDevice);
// з1
typedef unsigned short ushort;
struct ushort2 { unsigned short x,y; };
struct ushort4 { unsigned short x,y,z,w; };
struct float2 { float x,y; };
struct float4 { float x,y,z,w; };
struct int2 { int x,y; };
struct int4 { int x,y,z,w; };
__device__ float __uint_as_float(unsigned);
__device__ unsigned __float_as_uint(float);
__device__ extern const int warpSize;
__device__ float rsqrtf(float);
__device__ float __expf(float);
__device__ float __logf(float);
__device__ float fmaxf(float,float);
__device__ float fminf(float,float);
__device__ float __shfl_xor(float, int, int=64);
// о1
__device__ float powf(float,float);
__device__ float cosf(float);
__device__ float sinf(float);
__device__ float expf(float);
__device__ float logf(float);
__device__ float sqrtf(float);
__device__ float fabsf(float);
__device__ float tanhf(float);
__device__ float erff(float);
__device__ float floorf(float);
__device__ float roundf(float);
__device__ float fmodf(float,float);
__device__ double exp(double);
__device__ double log(double);
__device__ double sqrt(double);
__device__ double fabs(double);
__device__ bool isnan(float);
__device__ bool isinf(float);
template<class T> __device__ T max(T a, T b){ return a>b?a:b; }
template<class T> __device__ T min(T a, T b){ return a<b?a:b; }
__device__ float fmax(float,float);
__device__ float fmin(float,float);
__device__ double atomicAdd(double*, double);
__device__ float4 make_float4(float,float,float,float);
__device__ float2 make_float2(float,float);
__device__ float fmaf(float,float,float);
__device__ int printf(const char*, ...);
