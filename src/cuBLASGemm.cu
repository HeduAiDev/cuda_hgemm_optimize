#include <cublas_v2.h>
#include "gemm/base.hpp"
#include "utils/tensor.hpp"
#include "gemm.cuh"

gemm::base::GemmOutput cuBLASGemm(half *A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K )
{
    using namespace utils::tensor;
    utils::Timeit t;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // malloc on device
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    half alpha = static_cast<half>(1);
    half beta = static_cast<half>(0);

    t.start();
   cublasStatus_t status = cublasGemmEx(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 B.devicePtr(), CUDA_R_16F, N,
                                 A.devicePtr(), CUDA_R_16F, K,
                                 &beta,
                                 C.devicePtr(), CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    t.stop();
    gemm::base::GemmOutput options;
    options.status = gemm::base::Status::Success;
    options.excute_time_ms = t.elapsed_time;
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        options.status = gemm::base::Status::Error;
        options.code = (cudaError_t)status;
        options.err = "CuBlas error: " + std::string(cublasGetStatusString(status)) + "\n" + __FILE__ + ":" + std::to_string(__LINE__) + "\n";
    }
    C.copyToHost();
    cublasDestroy(handle);
    return options;
}