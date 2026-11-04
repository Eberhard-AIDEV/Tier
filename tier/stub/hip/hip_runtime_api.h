// а1
#pragma once

#include <cstddef>
#include <cstdint>

#include "hip/hip_vector_types.h"
#include "hip/library_types.h"

// б1
typedef enum hipError_t {
    hipSuccess                     = 0,
    hipErrorInvalidValue           = 1,
    hipErrorOutOfMemory            = 2,
    hipErrorMemoryAllocation       = 2,
    hipErrorNotInitialized         = 3,
    hipErrorInitializationError    = 3,
    hipErrorDeinitialized          = 4,
    hipErrorProfilerDisabled       = 5,
    hipErrorInvalidConfiguration   = 9,
    hipErrorInvalidSymbol          = 13,
    hipErrorInvalidDevicePointer   = 17,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInsufficientDriver     = 35,
    hipErrorNoDevice               = 100,
    hipErrorInvalidDevice          = 101,
    hipErrorInvalidImage           = 200,
    hipErrorInvalidContext         = 201,
    hipErrorMapFailed              = 205,
    hipErrorNoBinaryForGpu         = 209,
    hipErrorInvalidKernelFile      = 218,
    hipErrorInvalidHandle          = 400,
    hipErrorInvalidResourceHandle  = 400,
    hipErrorIllegalState           = 401,
    hipErrorNotFound               = 500,
    hipErrorNotReady               = 600,
    hipErrorIllegalAddress         = 700,
    hipErrorLaunchOutOfResources   = 701,
    hipErrorLaunchTimeOut          = 702,
    hipErrorLaunchFailure          = 719,
    hipErrorNotSupported           = 801,
    hipErrorUnknown                = 999
} hipError_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost     = 0,
    hipMemcpyHostToDevice   = 1,
    hipMemcpyDeviceToHost   = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault        = 4
} hipMemcpyKind;

// в1
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t*  hipEvent_t;
typedef int                  hipDevice_t;

#define hipStreamDefault     0x00u
#define hipStreamNonBlocking 0x01u

#define hipEventDefault       0x00u
#define hipEventBlockingSync  0x01u
#define hipEventDisableTiming 0x02u
#define hipEventInterprocess  0x04u

#define hipHostMallocDefault       0x00u
#define hipHostMallocPortable      0x01u
#define hipHostMallocMapped        0x02u
#define hipHostMallocWriteCombined 0x04u
#define hipHostMallocNumaUser      0x20000000u

#define hipHostRegisterDefault  0x00u
#define hipHostRegisterPortable 0x01u
#define hipHostRegisterMapped   0x02u

#define hipDeviceScheduleAuto 0x00u

// в2
#define HIP_SYMBOL(x) ((const void*)&(x))

// г1
typedef struct hipDeviceProp_t {
    char   name[256];
    char   gcnArchName[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    int    memoryClockRate;
    int    memoryBusWidth;
    size_t totalConstMem;
    int    major;
    int    minor;
    int    multiProcessorCount;
    int    l2CacheSize;
    int    maxThreadsPerMultiProcessor;
    int    computeMode;
    int    concurrentKernels;
    int    canMapHostMemory;
    int    integrated;
    int    managedMemory;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    isMultiGpuBoard;
    int    cooperativeLaunch;
    int    gcnArch;
} hipDeviceProp_t;

typedef struct hipFuncAttributes {
    int    binaryVersion;
    int    cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int    maxDynamicSharedSizeBytes;
    int    maxThreadsPerBlock;
    int    numRegs;
    int    preferredShmemCarveout;
    int    ptxVersion;
    size_t sharedSizeBytes;
} hipFuncAttributes;

// д1
hipError_t  hipGetLastError(void);
hipError_t  hipPeekAtLastError(void);
const char* hipGetErrorString(hipError_t hipError);
const char* hipGetErrorName(hipError_t hipError);

// д2
hipError_t hipGetDeviceCount(int* count);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipSetDevice(int deviceId);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);

// д3
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags = hipHostMallocDefault);
hipError_t hipHostFree(void* ptr);
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
hipError_t hipHostUnregister(void* hostPtr);
hipError_t hipMemGetInfo(size_t* free, size_t* total);

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream = 0);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, hipMemcpyKind kind,
                            hipStream_t stream = 0);
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                             size_t offset = 0, hipMemcpyKind kind = hipMemcpyHostToDevice);
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes,
                               size_t offset = 0, hipMemcpyKind kind = hipMemcpyDeviceToHost);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream = 0);

// д4
template <class T>
static inline hipError_t hipMalloc(T** devPtr, size_t size) {
    return hipMalloc(reinterpret_cast<void**>(devPtr), size);
}
template <class T>
static inline hipError_t hipHostMalloc(T** hostPtr, size_t size,
                                       unsigned int flags = hipHostMallocDefault) {
    return hipHostMalloc(reinterpret_cast<void**>(hostPtr), size, flags);
}

// ж1
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamQuery(hipStream_t stream);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags = 0);

// ж2
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

// ж3
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem = 0, hipStream_t stream = 0);
extern "C" hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                                size_t* sharedMem, hipStream_t* stream);
hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes = 0, hipStream_t stream = 0);
