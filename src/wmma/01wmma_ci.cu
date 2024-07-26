#include "gemm.cuh"
#include "utils/tensor.hpp"
#include <mma.h>
using namespace gemm::base;

#define BlockTileM (16 * 16)
#define BlockTileN (16 * 8)
#define BlockTileK (16)
#define WarpTileM (16 * 4)
#define WarpTileN (16 * 4)
#define WarpTileK 16

#define WarpSize 32
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16


// 0.523227 ms, M=N=2048, K=1024
__global__ void wmma_ci_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / (BlockTileN / WarpTileN);
    int warp_n = warp_id % (BlockTileN / WarpTileN);
    int offset_warp_st_global_cx = warp_n * WarpTileN;
    int offset_warp_st_global_cy = warp_m * WarpTileM;
    int offset_warp_ld_frag_a = offset_warp_st_global_cy;
    int offset_warp_ld_frag_b = offset_warp_st_global_cx;
    constexpr int frag_m_size = WarpTileM / WMMA_M;
    constexpr int frag_n_size = WarpTileN / WMMA_N;
    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN))
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag[frag_m_size];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag[frag_n_size];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[frag_m_size][frag_n_size];
    for (int i = 0; i < frag_m_size; i++) {
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::fill_fragment(c_frag[i][j], half(0.0f));
        }
    }

    for (int k = 0; k < K; k += WarpTileK) {
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++) {
            nvcuda::wmma::load_matrix_sync(a_frag[i], blockA_ptr + (offset_warp_ld_frag_a + i * WMMA_M) * K + k, K);
        }
        #pragma unroll
        for (int i = 0; i < frag_n_size; i++) {
            nvcuda::wmma::load_matrix_sync(b_frag[i], blockB_ptr + k * N + (offset_warp_ld_frag_b + i * WMMA_N), N);
        }
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++) {
            #pragma unroll
            for (int j = 0; j < frag_n_size; j++) {
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }
    __syncthreads();
    for (int i = 0; i < frag_m_size; i++) {
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::store_matrix_sync(blockC_ptr + (offset_warp_st_global_cy + i * WMMA_M) * N + offset_warp_st_global_cx + j * WMMA_N, c_frag[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
};




gemm::base::GemmOutput wmma_ci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
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
        wmma_ci_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
