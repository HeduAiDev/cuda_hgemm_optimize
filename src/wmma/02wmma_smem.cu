#include "gemm.cuh"
#include "utils/tensor.hpp"
#include <mma.h>
using namespace gemm::base;

#define BlockTileM (16 * 16)
#define BlockTileN (16 * 16)
#define BlockTileK 16
#define WarpTileM (16 * 4)
#define WarpTileN (16 * 4)
#define WarpTileK 16

#define WarpSize 32
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16


// 0.420666 ms, M=N=2048, K=1024
__global__ void wmma_smem_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / (BlockTileN / WarpTileN);
    int warp_n = warp_id % (BlockTileN / WarpTileN);
    int offset_warp_st_global_cx = warp_n * WarpTileN;
    int offset_warp_st_global_cy = warp_m * WarpTileM;
    int offset_warp_ld_frag_a = offset_warp_st_global_cy;
    int offset_warp_ld_frag_b = offset_warp_st_global_cx;
    constexpr int float4_element_num = 8;
    constexpr int frag_m_size = WarpTileM / WMMA_M;
    constexpr int frag_n_size = WarpTileN / WMMA_N;
    constexpr int ldm_blockA_f4size = BlockTileK / float4_element_num;
    constexpr int ldm_blockB_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_blockA = BlockTileK;
    constexpr int ldm_blockB = BlockTileN;
    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN))
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))

    __shared__ half smem_A[BlockTileM * BlockTileK];
    __shared__ half smem_B[BlockTileK * BlockTileN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag[frag_m_size];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag[frag_n_size];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[frag_m_size][frag_n_size];
    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::fill_fragment(c_frag[i][j], half(0.0f));
        }
    }
    for (int k = 0; k < K; k += BlockTileK)
    {
        // load data from global to shared memory
        #pragma unroll
        for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += blockDim.x)
        {
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;
            int offset_ld2s_global_by = i / ldm_blockA_f4size;
            reinterpret_cast<float4 *>(smem_A)[offset_ld2s_global_by * ldm_blockA_f4size + offset_ld2s_global_bx] = *(reinterpret_cast<float4 *>(blockA_ptr + offset_ld2s_global_by * K + offset_ld2s_global_bx * float4_element_num + k));
        }
        #pragma unroll
        for (int i = tid; i < BlockTileK * BlockTileN / float4_element_num; i += blockDim.x)
        {
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;
            int offset_ld2s_global_by = i / ldm_blockB_f4size;
            reinterpret_cast<float4 *>(smem_B)[offset_ld2s_global_by * ldm_blockB_f4size + offset_ld2s_global_bx] = *(reinterpret_cast<float4 *>(blockB_ptr + (offset_ld2s_global_by + k) * N + offset_ld2s_global_bx * float4_element_num));
        }
        __syncthreads();

        #pragma unroll
        for (int bk = 0; bk < BlockTileK; bk += WarpTileK)
        {
            #pragma unroll
            for (int i = 0; i < frag_m_size; i++) {
                nvcuda::wmma::load_matrix_sync(a_frag[i], smem_A + (offset_warp_ld_frag_a + i * WMMA_M) * ldm_blockA + bk, ldm_blockA);
            }
            #pragma unroll
            for (int i = 0; i < frag_n_size; i++) {
                nvcuda::wmma::load_matrix_sync(b_frag[i], smem_B + bk * ldm_blockB + (offset_warp_ld_frag_b + i * WMMA_N), ldm_blockB);
            }
            #pragma unroll
            for (int i = 0; i < frag_m_size; i++)
            {
                #pragma unroll
                for (int j = 0; j < frag_n_size; j++)
                {
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::store_matrix_sync(blockC_ptr + (offset_warp_st_global_cy + i * WMMA_M) * N + offset_warp_st_global_cx + j * WMMA_N, c_frag[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
};




gemm::base::GemmOutput wmma_smem(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
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
        wmma_smem_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
