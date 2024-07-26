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
// reference: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/#software_pipelining
// ←-----------------------------------------------------------------------------------
// ⤷---------------------------------------iter k-----------------------------------→-⤴
// |████████████████load global███████████████████████|███store shared███|             |  Global to Shared Memory
// |---------------------------------------iter bk-----------------------↘-------------|
// |█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|  Shared Memory to Registers
// ↘-------------↘------------↘-------------↘-------------↘-------------↘-------------↘
// |████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|  Registers to CUDA cores

#define LOAD_GLOBAL(_K)                                                                                                                                                               \
    {                                                                                                                                                                                 \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += total_threads)                                                                     \
        {                                                                                                                                                                             \
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;                                                                                                                        \
            int offset_ld2s_global_by = i / ldm_blockA_f4size;                                                                                                                        \
            buffer_a[i / total_threads] = *(reinterpret_cast<float4 *>(blockA_ptr + offset_ld2s_global_by * K + offset_ld2s_global_bx * float4_element_num + (_K) * BlockTileK));   \
        }                                                                                                                                                                             \
        _Pragma("unroll") for (int i = tid; i < BlockTileK * BlockTileN / float4_element_num; i += total_threads)                                                                     \
        {                                                                                                                                                                             \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                                        \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                                        \
            buffer_b[i / total_threads] = *(reinterpret_cast<float4 *>(blockB_ptr + (offset_ld2s_global_by + (_K) * BlockTileK) * N + offset_ld2s_global_bx * float4_element_num)); \
        }                                                                                                                                                                             \
    }

#define STORE_SHARED(SMEM_WRITE_IDX)                                                                                                                                  \
    {                                                                                                                                                                 \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += total_threads)                                                     \
        {                                                                                                                                                             \
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;                                                                                                        \
            int offset_ld2s_global_by = i / ldm_blockA_f4size;                                                                                                        \
            reinterpret_cast<float4 *>(smem_A + (SMEM_WRITE_IDX))[offset_ld2s_global_by * ldm_blockA_f4size + offset_ld2s_global_bx] = buffer_a[i / total_threads]; \
        }                                                                                                                                                             \
        _Pragma("unroll") for (int i = tid; i < BlockTileK * BlockTileN / float4_element_num; i += total_threads)                                                     \
        {                                                                                                                                                             \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                        \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                        \
            reinterpret_cast<float4 *>(smem_B + (SMEM_WRITE_IDX))[offset_ld2s_global_by * ldm_blockB_f4size + offset_ld2s_global_bx] = buffer_b[i / total_threads]; \
        }                                                                                                                                                             \
    }

#define LOAD_SHARED(BK, SMEM_READ_IDX, REG_WRITE_IDX)                                                                                                                                                              \
    {                                                                                                                                                                                                              \
        _Pragma("unroll") for (int i = 0; i < frag_m_size; i++)                                                                                                                                                    \
        {                                                                                                                                                                                                          \
            nvcuda::wmma::load_matrix_sync(a_frag[(REG_WRITE_IDX)][i], reinterpret_cast<half *>(smem_A + (SMEM_READ_IDX)) + (offset_warp_ld_frag_a + i * WMMA_M) * ldm_blockA + (BK) * WarpTileK, ldm_blockA);     \
            _Pragma("unroll") for (int j = 0; j < frag_n_size; j++)                                                                                                                                                \
            {                                                                                                                                                                                                      \
                nvcuda::wmma::load_matrix_sync(b_frag[(REG_WRITE_IDX)][j], reinterpret_cast<half *>(smem_B + (SMEM_READ_IDX)) + (BK) * WarpTileK * ldm_blockB + (offset_warp_ld_frag_b + j * WMMA_N), ldm_blockB); \
            }                                                                                                                                                                                                      \
        }                                                                                                                                                                                                          \
    }

// 0.510772 ms, M=N=2048, K=1024
__global__ void wmma_pipline_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
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
    constexpr int total_threads = (BlockTileN / WarpTileN) * (BlockTileM / WarpTileM) * WarpSize;
    constexpr int buffer_a_size = BlockTileM * BlockTileK / float4_element_num / total_threads;
    constexpr int buffer_b_size = BlockTileN * BlockTileK / float4_element_num / total_threads;
    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN))
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))

    __shared__ half smem_A[2][BlockTileM * BlockTileK];
    __shared__ half smem_B[2][BlockTileK * BlockTileN];
    float4 buffer_a[buffer_a_size];
    float4 buffer_b[buffer_b_size];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag[2][frag_m_size];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag[2][frag_n_size];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[frag_m_size][frag_n_size];
    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::fill_fragment(c_frag[i][j], half(0.0f));
        }
    }

    bool smem_write_idx = 0;
    bool reg_write_idx = 0;
    LOAD_GLOBAL(0);
    STORE_SHARED(0);
    __syncthreads();
    // load smem first bk data
    LOAD_SHARED(0, 0, 0);
    // handle [0:-1] gmem blocks and load [1:] gmem blocks
    for (int k = 1; k < K / BlockTileK; k++)
    {
        smem_write_idx = !smem_write_idx;
        // load next gmem block
        LOAD_GLOBAL(k) 

        // handle [0:-1] smem blocks and load [1:] smem blocks
        #pragma unroll
        for (int bk = 1; bk < BlockTileK / WarpTileK; bk++)
        {
            reg_write_idx = !reg_write_idx;
            // load next bk
            LOAD_SHARED(bk, !smem_write_idx, reg_write_idx);
            // compute
            #pragma unroll
            for (int i = 0; i < frag_m_size; i++)
            {
                #pragma unroll
                for (int j = 0; j < frag_n_size; j++)
                {
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[!reg_write_idx][i], b_frag[!reg_write_idx][j], c_frag[i][j]);
                }
            }
        }
        // handle last smem block, different from above, register load next smem block's first bk
        // overlap store shared and compute
        // store gmem block from buffer to smem
        STORE_SHARED(smem_write_idx);
        // compute
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++)
        {
            #pragma unroll
            for (int j = 0; j < frag_n_size; j++)
            {
                // reg_write_idx is last smem block load in above last loop
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[reg_write_idx][i], b_frag[reg_write_idx][j], c_frag[i][j]);
            }
        }
        __syncthreads();
        // load smem first bk data 
        LOAD_SHARED(0, smem_write_idx, reg_write_idx);
    }
    // handle last gmem block, different from above there is no next gmem block to load
    smem_write_idx = !smem_write_idx;
    // handle all smem blocks and load [1:] smem blocks
    #pragma unroll
    for (int bk = 0; bk < BlockTileK / WarpTileK; bk++)
    {
        reg_write_idx = !reg_write_idx;
        // load next bk
        if (bk < BlockTileK / WarpTileK - 1) {
            LOAD_SHARED(bk + 1, !smem_write_idx, reg_write_idx);
        }
        // compute
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++)
        {
            #pragma unroll
            for (int j = 0; j < frag_n_size; j++)
            {
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[!reg_write_idx][i], b_frag[!reg_write_idx][j], c_frag[i][j]);
            }
        }
    }


    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            nvcuda::wmma::store_matrix_sync(blockC_ptr + (offset_warp_st_global_cy + i * WMMA_M) * N + offset_warp_st_global_cx + j * WMMA_N, c_frag[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
};




gemm::base::GemmOutput wmma_pipline(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
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
        wmma_pipline_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
