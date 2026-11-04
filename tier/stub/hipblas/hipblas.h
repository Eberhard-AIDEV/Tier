#pragma once
#include <hip/hip_runtime.h>
typedef enum { HIPBLAS_STATUS_SUCCESS=0, HIPBLAS_STATUS_NOT_SUPPORTED=10 } hipblasStatus_t;
typedef enum { HIPBLAS_OP_N=111, HIPBLAS_OP_T=112 } hipblasOperation_t;
typedef enum { HIP_R_32F=0, HIP_R_16BF=1, HIP_R_16F=2 } hipDataType;
typedef struct ihipblasHandle_t* hipblasHandle_t;
hipblasStatus_t hipblasCreate(hipblasHandle_t*);
hipblasStatus_t hipblasDestroy(hipblasHandle_t);
hipblasStatus_t hipblasSetStream(hipblasHandle_t, hipStream_t);
hipblasStatus_t hipblasSgemm(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                             int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
