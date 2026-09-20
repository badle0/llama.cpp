#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

// Minimal C ABI used by the Denglin provider. The SDK release links
// these symbols from libdlblas but does not ship the cublas_v2.h dependency of
// dlblas_ext.h, so importing that public header makes an otherwise valid SDK
// impossible to consume. Keep vendor types out of the generic backend layer.
typedef struct cublasContext * cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef int cudaDataType_t;

static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
static constexpr cublasOperation_t CUBLAS_OP_N = 0;
static constexpr cublasOperation_t CUBLAS_OP_T = 1;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_32F = 68;
static constexpr cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT = -1;
static constexpr cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;
static constexpr cudaDataType_t CUDA_R_32F = 0;
static constexpr cudaDataType_t CUDA_R_16F = 2;
static constexpr cudaDataType_t CUDA_R_4U = 18;

struct flagos_dlblas_quant_parameters_v2 {
    int32_t a_group_size_m;
    int32_t a_group_size_k;
    void * a_zeropoints;
    cudaDataType_t a_zeropoints_type;
    void * a_scales;
    cudaDataType_t a_scales_type;

    int32_t b_group_size_k;
    int32_t b_group_size_n;
    void * b_zeropoints;
    cudaDataType_t b_zeropoints_type;
    void * b_scales;
    cudaDataType_t b_scales_type;

    int32_t c_group_size_m;
    int32_t c_group_size_n;
    void * c_zeropoints;
    cudaDataType_t c_zeropoints_type;
    void * c_scales;
    cudaDataType_t c_scales_type;
};

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t * handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream);

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void * alpha,
    const void * a,
    cudaDataType_t atype,
    int lda,
    const void * b,
    cudaDataType_t btype,
    int ldb,
    const void * beta,
    void * c,
    cudaDataType_t ctype,
    int ldc,
    cublasComputeType_t compute_type,
    cublasGemmAlgo_t algo);

cublasStatus_t dlblasGemmExV2(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void * alpha,
    const void * a,
    cudaDataType_t atype,
    int lda,
    const void * b,
    cudaDataType_t btype,
    int ldb,
    const void * beta,
    void * c,
    cudaDataType_t ctype,
    int ldc,
    cudaDataType_t compute_type,
    cublasGemmAlgo_t algo,
    flagos_dlblas_quant_parameters_v2 * quant_parameters);

}

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSetStream cublasSetStream_v2
