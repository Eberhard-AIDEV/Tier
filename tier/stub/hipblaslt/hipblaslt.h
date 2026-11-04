#pragma once
#include <hipblas/hipblas.h>
typedef struct ihipblasLtHandle_t* hipblasLtHandle_t;
typedef struct ihipblasLtMatmulDesc_t* hipblasLtMatmulDesc_t;
typedef struct ihipblasLtMatrixLayout_t* hipblasLtMatrixLayout_t;
typedef struct ihipblasLtMatmulPreference_t* hipblasLtMatmulPreference_t;
struct hipblasLtMatmulAlgo_t { char data[64]; };
struct hipblasLtMatmulHeuristicResult_t { hipblasLtMatmulAlgo_t algo; size_t workspaceSize; hipblasStatus_t state; float wavesCount; int reserved[4]; };
typedef enum { HIPBLASLT_MATMUL_DESC_TRANSA=0, HIPBLASLT_MATMUL_DESC_TRANSB=1 } hipblasLtMatmulDescAttributes_t;
typedef enum { HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=0 } hipblasLtMatmulPreferenceAttributes_t;
typedef enum { HIPBLAS_COMPUTE_32F=0 } hipblasComputeType_t;
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t*);
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t*, hipblasComputeType_t, hipDataType);
hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t);
hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t, const void*, size_t);
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t*, hipDataType, uint64_t, uint64_t, int64_t);
hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t);
hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t*);
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t);
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceAttributes_t, const void*, size_t);
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                                                hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatmulPreference_t, int,
                                                hipblasLtMatmulHeuristicResult_t*, int*);
hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*, const void*, hipblasLtMatrixLayout_t,
                                const void*, hipblasLtMatrixLayout_t, const void*, const void*, hipblasLtMatrixLayout_t,
                                void*, hipblasLtMatrixLayout_t, const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t);
