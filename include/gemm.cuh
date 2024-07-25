#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gemm/base.hpp"

gemm::base::GemmOutput cuBLASGemm(half *A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K );

// class SimtNaiveOptions: public gemm::base::GemmOptions {
//     public:
//         SimtNaiveOptions() {
//             this -> add("BlockTileM", 16);
//             this -> add("BlockTileN", 16);
//         }
// };

gemm::base::GemmOutput simt_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput simt_regci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput simt_smem(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput simt_smemT(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput simt_pipline(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);

//wmma
gemm::base::GemmOutput wmma_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput wmma_ci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput wmma_smem(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput wmma_pipline(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);

//mma
gemm::base::GemmOutput mma_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput mma_ci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput mma_smem(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);
gemm::base::GemmOutput mma_ldmatrix(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);