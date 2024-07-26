#include "gemm.cuh"
#include "utils/tensor.hpp"
#include <mma.h>
#include <cstdint>
using namespace gemm::base;

#define BlockTileM (16 * 4)
#define BlockTileN (8 * 2)

#define MMA_M  16
#define MMA_N  8
#define MMA_K  8

#define WarpSize 32


// 2.444602 ms, M=N=2048, K=1024
__global__ void mma_naive_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / (BlockTileN / MMA_N);
    int warp_n = warp_id % (BlockTileN / MMA_N);
    int offset_warp_st_global_cx = warp_n * MMA_N;
    int offset_warp_st_global_cy = warp_m * MMA_M;
    int offset_warp_ld_frag_a = offset_warp_st_global_cy;
    int offset_warp_ld_frag_b = offset_warp_st_global_cx;
    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN) * K)
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))
    #define warpA_ptr (blockA_ptr + offset_warp_ld_frag_a * K)
    #define warpB_ptr (blockB_ptr + offset_warp_ld_frag_b * K)
    #define warpC_ptr (blockC_ptr + offset_warp_st_global_cy * N + offset_warp_st_global_cx)
    // r: u32
    // h: u16
    // l: u64
    // f: fp32
    // d: fp64
    #define _u32(x) (*((uint32_t*)(x)))
    // atomic block is 8x8 size, reg num in a row is 8/2 = 4
    int offset_thread_cx = (lane_id % 4) * 2;
    int offset_thread_cy = lane_id / 4;

    uint32_t regD[2] = {0,0};
    

    for (int k = 0; k < K; k += MMA_K) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3}, "
            "{%4}, "
            "{%5, %6}; "
            :"=r"(regD[0]), "=r"(regD[1])
            :"r"(_u32(warpA_ptr + offset_thread_cy * K + k + offset_thread_cx)), "r"(_u32(warpA_ptr + (offset_thread_cy + 8) * K + k + offset_thread_cx)),
             "r"(_u32(warpB_ptr + offset_thread_cy * K + k + offset_thread_cx)),
             "r"(regD[0]), "r"(regD[1])
        );
    }
    _u32(warpC_ptr + offset_thread_cy * N + offset_thread_cx) = regD[0];
    _u32(warpC_ptr + (offset_thread_cy + 8) * N + offset_thread_cx) = regD[1];
};




gemm::base::GemmOutput mma_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    B.transpose();
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    assert(BlockTileM % MMA_M == 0 && BlockTileN % MMA_N == 0 && MMA_K % 8 == 0);
    assert(MMA_M % 8 == 0 && MMA_N % 8 == 0);

    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block((BlockTileN / MMA_N) * (BlockTileM / MMA_M) * WarpSize);
    utils::Timeit t;
    for (int i = 0; i < launch_times; i++) {
        t.start();
        mma_naive_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};