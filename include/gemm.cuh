#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gemm/base.hpp"

gemm::base::GemmOutput cuBLASGemm(half *A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K );

class SimtNaiveOptions: public gemm::base::GemmOptions {
    public:
        SimtNaiveOptions() {
            this -> add("BlockTileM", 16);
            this -> add("BlockTileN", 16);
        }
};

gemm::base::GemmOutput simt_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);


class SimtRegCIOptions: public gemm::base::GemmOptions {
    public:
        SimtRegCIOptions() {
            this -> add("BlockTileM", 128);
            this -> add("BlockTileN", 128);
            this -> add("ThreadTileM", 8);
            this -> add("ThreadTileN", 8);
        }
};

gemm::base::GemmOutput simt_regci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times = 1);