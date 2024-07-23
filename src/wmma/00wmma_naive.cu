#include "gemm.cuh"
#include "utils/tensor.hpp"
#include <mma.h>
using namespace gemm::base;

#define BlockTileM (16 * 4)
#define BlockTileN (16 * 2)
#define BlockTileK (16)
#define WarpTileM 16
#define WarpTileN 16
#define WarpTileK 16

#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

#define WarpSize 32


// 1.734089 ms, M=N=2048, K=1024
__global__ void wmma_naive_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / (BlockTileN / WarpTileN);
    int warp_n = warp_id % (BlockTileN / WarpTileN);
    int offset_warp_st_global_cx = warp_n * WarpTileN;
    int offset_warp_st_global_cy = warp_m * WarpTileM;
    int offset_warp_ld_frag_a = offset_warp_st_global_cy;
    int offset_warp_ld_frag_b = offset_warp_st_global_cx;
    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN))
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, half(0.0f));

    for (int k = 0; k < K; k += WarpTileK) {
        nvcuda::wmma::load_matrix_sync(a_frag, blockA_ptr + offset_warp_ld_frag_a * K + k, K);
        nvcuda::wmma::load_matrix_sync(b_frag, blockB_ptr + k * N + offset_warp_ld_frag_b, N);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
    nvcuda::wmma::store_matrix_sync(blockC_ptr + offset_warp_st_global_cy * N + offset_warp_st_global_cx, c_frag, N, nvcuda::wmma::mem_row_major);
};




gemm::base::GemmOutput wmma_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block((BlockTileN / WarpTileN) * (BlockTileM / WarpTileM) * WarpSize);
    utils::Timeit t;
    for (int i = 0; i < launch_times; i++) {
        t.start();
        wmma_naive_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
