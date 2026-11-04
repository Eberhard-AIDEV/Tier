#pragma once
#include <cublas_v2.h>
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque* cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque* cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulPreferenceOpaque* cublasLtMatmulPreference_t;
struct cublasLtMatmulAlgo_t { char data[64]; };
struct cublasLtMatmulHeuristicResult_t { cublasLtMatmulAlgo_t algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4]; };
typedef enum { CUBLASLT_MATMUL_DESC_TRANSA=0, CUBLASLT_MATMUL_DESC_TRANSB=1 } cublasLtMatmulDescAttributes_t;
typedef enum { CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES=0 } cublasLtMatmulPreferenceAttributes_t;
typedef enum { CUBLAS_COMPUTE_32F=0 } cublasComputeType_t;
cublasStatus_t cublasLtCreate(cublasLtHandle_t*);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t);
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                              cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int,
                                              cublasLtMatmulHeuristicResult_t*, int*);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*, cublasLtMatrixLayout_t,
                              const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
                              void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);
