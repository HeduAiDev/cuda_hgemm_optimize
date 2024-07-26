#include "gemm.cuh"
#include "utils/tensor.hpp"
#include <mma.h>
#include <cstdint>
using namespace gemm::base;

#define BlockTileM (16 * 16)
#define BlockTileN (16 * 8)
#define BlockTileK 16

#define WarpTileM (16 * 4)
#define WarpTileN (16 * 4)
#define WarpTileK 8

#define WarpSize 32
#define MMA_M  16
#define MMA_N  8
#define MMA_K  8
// reference: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/#software_pipelining
// ←-----------------------------------------------------------------------------------
// ⤷---------------------------------------iter k-----------------------------------→-⤴
// |████████████████load global███████████████████████|███store shared███|             |  Global to Shared Memory
// |---------------------------------------iter bk-----------------------↘-------------|
// |█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|  Shared Memory to Registers
// ↘-------------↘------------↘-------------↘-------------↘-------------↘-------------↘
// |████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|  Registers to CUDA cores

#define LOAD_GLOBAL(_K)                                                                                                                                                                  \
    {                                                                                                                                                                                    \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += total_threads)                                                                        \
        {                                                                                                                                                                                \
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;                                                                                                                           \
            int offset_ld2s_global_by = i / ldm_blockA_f4size;                                                                                                                           \
            buffer_a[i / total_threads] = reinterpret_cast<float4 *>(blockA_ptr)[offset_ld2s_global_by * ldm_A_f4size + offset_ld2s_global_bx + (_K) * BlockTileK / float4_element_num]; \
        }                                                                                                                                                                                \
        _Pragma("unroll") for (int i = tid; i < BlockTileN * BlockTileK / float4_element_num; i += total_threads)                                                                        \
        {                                                                                                                                                                                \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                                           \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                                           \
            buffer_b[i / total_threads] = reinterpret_cast<float4 *>(blockB_ptr)[offset_ld2s_global_by * ldm_B_f4size + offset_ld2s_global_bx + (_K) * BlockTileK / float4_element_num]; \
        }                                                                                                                                                                                \
    }

#define STORE_SHARED(SMEM_WRITE_IDX)                                                                                                                                \
    {                                                                                                                                                               \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += total_threads)                                                   \
        {                                                                                                                                                           \
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;                                                                                                      \
            int offset_ld2s_global_by = i / ldm_blockA_f4size;                                                                                                      \
            reinterpret_cast<float4 *>(smem_A + (SMEM_WRITE_IDX))[offset_ld2s_global_by * ldm_blockA_f4size + offset_ld2s_global_bx] = buffer_a[i / total_threads]; \
        }                                                                                                                                                           \
        _Pragma("unroll") for (int i = tid; i < BlockTileN * BlockTileK / float4_element_num; i += total_threads)                                                   \
        {                                                                                                                                                           \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                      \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                      \
            reinterpret_cast<float4 *>(smem_B + (SMEM_WRITE_IDX))[offset_ld2s_global_by * ldm_blockB_f4size + offset_ld2s_global_bx] = buffer_b[i / total_threads]; \
        }                                                                                                                                                           \
    }

#define LOAD_SHARED(BK, SMEM_READ_IDX, REG_WRITE_IDX)                                                                 \
    {                                                                                                                 \
        _Pragma("unroll") for (int i = 0; i < frag_m_size; i++)                                                       \
        {                                                                                                             \
            asm volatile(                                                                                             \
                "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"                                                   \
                : "=r"(regA[(REG_WRITE_IDX)][i][0]), "=r"(regA[(REG_WRITE_IDX)][i][1])                                \
                : "l"(smem_warpA_ptr((SMEM_READ_IDX)) + (i * MMA_M + lane_id % 16) * ldm_blockA + (BK) * WarpTileK)); \
        }                                                                                                             \
        _Pragma("unroll") for (int i = 0; i < frag_n_size; i++)                                                       \
        {                                                                                                             \
            asm volatile(                                                                                             \
                "ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"                                                       \
                : "=r"(regB[(REG_WRITE_IDX)][i])                                                                      \
                : "l"(smem_warpB_ptr((SMEM_READ_IDX)) + (i * MMA_N + lane_id % 8) * ldm_blockB + (BK) * WarpTileK));  \
        }                                                                                                             \
    }

// 0.410984 ms, M=N=2048, K=1024
__global__ void mma_pipline_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / (BlockTileN / WarpTileN);
    int warp_n = warp_id % (BlockTileN / WarpTileN);
    int offset_warp_st_global_cx = warp_n * WarpTileN;
    int offset_warp_st_global_cy = warp_m * WarpTileM;
    int offset_warp_ld_frag_a = offset_warp_st_global_cy;
    int offset_warp_ld_frag_b = offset_warp_st_global_cx;
    constexpr int float4_element_num = 8;
    constexpr int frag_m_size = WarpTileM / MMA_M;
    constexpr int frag_n_size = WarpTileN / MMA_N;
    constexpr int ldm_blockA_f4size = BlockTileK / float4_element_num;
    constexpr int ldm_blockB_f4size = BlockTileK / float4_element_num;
    constexpr int ldm_blockA = BlockTileK;
    constexpr int ldm_blockB = BlockTileK;
    constexpr int total_threads = (BlockTileN / WarpTileN) * (BlockTileM / WarpTileM) * WarpSize;
    constexpr int buffer_a_size = BlockTileM * BlockTileK / float4_element_num / total_threads;
    constexpr int buffer_b_size = BlockTileN * BlockTileK / float4_element_num / total_threads;

    int ldm_A = K;
    int ldm_B = K;
    int ldm_C = N;
    int ldm_A_f4size = K / float4_element_num;
    int ldm_B_f4size = K / float4_element_num;
    int ldm_C_f4size = N / float4_element_num;

    __shared__ half smem_A[2][BlockTileM * BlockTileK];
    __shared__ half smem_B[2][BlockTileN * BlockTileK];
    float4 buffer_a[buffer_a_size];
    float4 buffer_b[buffer_b_size];

    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN) * K)
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))
    #define smem_warpA_ptr(idx) (smem_A[(idx)] + offset_warp_ld_frag_a * ldm_blockA)
    #define smem_warpB_ptr(idx) (smem_B[(idx)] + offset_warp_ld_frag_b * ldm_blockB)
    #define gmem_warpC_ptr (blockC_ptr + offset_warp_st_global_cy * ldm_C + offset_warp_st_global_cx)


    // r: u32
    // r: u32
    // h: u16
    // l: u64
    // f: fp32
    // d: fp64
    #define _u32(x) (*((uint32_t*)(x)))

    // atomic block is 8x8 size, reg num in a row is 8/2 = 4
    int offset_thread_cx = (lane_id % 4) * 2;
    int offset_thread_cy = lane_id / 4;

    uint32_t regD[frag_m_size][frag_n_size][2];
    uint32_t regA[2][frag_m_size][2];
    uint32_t regB[2][frag_n_size];
    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            regD[i][j][0] = 0;
            regD[i][j][1] = 0;
        }
    }

    bool smem_write_idx = 0;
    bool reg_write_idx = 0;
    // global to shared memory
    LOAD_GLOBAL(0)
    STORE_SHARED(0)
    __syncthreads();
    // load first smem block
    LOAD_SHARED(0, 0, 0)
    // handle [0:-1] gmem blocks and load [1:] gmem blocks
    for (int k = 1; k < K / BlockTileK; k ++ ) {
        smem_write_idx = !smem_write_idx;
        // load next gmem block
        LOAD_GLOBAL(k)

        // handle [0:-1] smem blocks and load [1:] smem blocks
        #pragma unroll
        for (int bk = 1; bk < BlockTileK / WarpTileK; bk++ ) {
            reg_write_idx = !reg_write_idx;
            // load next smem block
            LOAD_SHARED(bk, !smem_write_idx, reg_write_idx)
            // compute
            #pragma unroll
            for (int i = 0; i < frag_m_size; i++) {
                #pragma unroll
                for (int j = 0; j < frag_n_size; j++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                        "{%0, %1}, "
                        "{%2, %3}, "
                        "{%4}, "
                        "{%5, %6}; "
                        :"=r"(regD[i][j][0]), "=r"(regD[i][j][1])
                        :"r"(regA[!reg_write_idx][i][0]), "r"(regA[!reg_write_idx][i][1]),
                         "r"(regB[!reg_write_idx][j]),
                         "r"(regD[i][j][0]), "r"(regD[i][j][1])
                    );
                }
            }
        }
        // handle last smem block, different from above, register load next smem block's first bk
        // overlap store shared and compute
        // store gmem block from buffer to smem
        STORE_SHARED(smem_write_idx)
        // compute
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++) {
            #pragma unroll
            for (int j = 0; j < frag_n_size; j++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, "
                    "{%2, %3}, "
                    "{%4}, "
                    "{%5, %6}; "
                    :"=r"(regD[i][j][0]), "=r"(regD[i][j][1])
                    :"r"(regA[reg_write_idx][i][0]), "r"(regA[reg_write_idx][i][1]),
                     "r"(regB[reg_write_idx][j]),
                     "r"(regD[i][j][0]), "r"(regD[i][j][1])
                );
            }
        }
        __syncthreads();
        // load first smem block
        LOAD_SHARED(0, smem_write_idx, !reg_write_idx)
        reg_write_idx = !reg_write_idx;
    }
    // handle last gmem block, different from above there is no next gmem block to load
    smem_write_idx = !smem_write_idx;
    // handle all smem blocks and load [1:] smem blocks
    #pragma unroll
    for (int bk = 0; bk < BlockTileK / WarpTileK; bk ++ ) {
        reg_write_idx = !reg_write_idx;
        if ( bk < BlockTileK / WarpTileK - 1) {
            LOAD_SHARED(bk + 1, !smem_write_idx, reg_write_idx)
        }
        // compute
        #pragma unroll
        for (int i = 0; i < frag_m_size; i++) {
            #pragma unroll
            for (int j = 0; j < frag_n_size; j++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, "
                    "{%2, %3}, "
                    "{%4}, "
                    "{%5, %6}; "
                    :"=r"(regD[i][j][0]), "=r"(regD[i][j][1])
                    :"r"(regA[!reg_write_idx][i][0]), "r"(regA[!reg_write_idx][i][1]),
                     "r"(regB[!reg_write_idx][j]),
                     "r"(regD[i][j][0]), "r"(regD[i][j][1])
                );
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            _u32(gmem_warpC_ptr + (offset_thread_cy + i * MMA_M) * ldm_C + offset_thread_cx + j * MMA_N) = regD[i][j][0];
            _u32(gmem_warpC_ptr + (offset_thread_cy + 8 + i * MMA_M) * ldm_C + offset_thread_cx + j * MMA_N) = regD[i][j][1];
        }
    }
};




gemm::base::GemmOutput mma_pipline(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    B.transpose();
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    assert(BlockTileM % WarpTileM == 0 && BlockTileN % WarpTileN == 0 && WarpTileK == MMA_K);
    assert(WarpTileM % MMA_M == 0 && WarpTileN % MMA_N == 0);
    assert(MMA_M % 8 == 0 && MMA_N % 8 == 0 && MMA_K % 8 == 0);

    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block((BlockTileN / WarpTileN) * (BlockTileM / WarpTileM) * WarpSize);
    utils::Timeit t;
    for (int i = 0; i < launch_times; i++) {
        t.start();
        mma_pipline_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
