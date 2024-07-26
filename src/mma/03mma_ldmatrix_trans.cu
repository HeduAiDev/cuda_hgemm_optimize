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

// this version use ldmatrix trans to transpose B
// 0.470435 ms, M=N=2048, K=1024
__global__ void mma_ldmatrix_trans_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
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
    constexpr int ldm_blockB_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_blockA = BlockTileK;
    constexpr int ldm_blockB = BlockTileN;

    int ldm_A = K;
    int ldm_B = N;
    int ldm_C = N;
    int ldm_A_f4size = K / float4_element_num;
    int ldm_B_f4size = N / float4_element_num;
    int ldm_C_f4size = N / float4_element_num;

    __shared__ half smem_A[BlockTileM * BlockTileK];
    __shared__ half smem_B[BlockTileN * BlockTileK];

    #define blockA_ptr (A + (blockIdx.y * BlockTileM) * K)
    #define blockB_ptr (B + (blockIdx.x * BlockTileN))
    #define blockC_ptr (C + (blockIdx.y * BlockTileM) * N + (blockIdx.x * BlockTileN))
    #define smem_warpA_ptr (smem_A + offset_warp_ld_frag_a * ldm_blockA)
    #define smem_warpB_ptr (smem_B + offset_warp_ld_frag_b)
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
    uint32_t regA[frag_m_size][2];
    uint32_t regB[frag_n_size];
    #pragma unroll
    for (int i = 0; i < frag_m_size; i++) {
        #pragma unroll
        for (int j = 0; j < frag_n_size; j++) {
            regD[i][j][0] = 0;
            regD[i][j][1] = 0;
        }
    }

    for (int k = 0; k < K; k += BlockTileK) {
        // from global memory to shared memory
        #pragma unroll
        for (int i = tid; i < BlockTileM * BlockTileK / float4_element_num; i += blockDim.x) {
            int offset_ld2s_global_bx = i % ldm_blockA_f4size;
            int offset_ld2s_global_by = i / ldm_blockA_f4size;
            float4 buffer = reinterpret_cast<float4*>(blockA_ptr)[offset_ld2s_global_by * ldm_A_f4size + offset_ld2s_global_bx + k / float4_element_num];
            reinterpret_cast<float4*>(smem_A)[offset_ld2s_global_by * ldm_blockA_f4size + offset_ld2s_global_bx] = buffer;
        }
        #pragma unroll
        for (int i = tid; i < BlockTileN * BlockTileK / float4_element_num; i += blockDim.x) {
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;
            int offset_ld2s_global_by = i / ldm_blockB_f4size;
            float4 buffer = reinterpret_cast<float4*>(blockB_ptr)[(offset_ld2s_global_by + k) * ldm_B_f4size + offset_ld2s_global_bx];
            reinterpret_cast<float4*>(smem_B)[offset_ld2s_global_by * ldm_blockB_f4size + offset_ld2s_global_bx] = buffer;
        }
        __syncthreads();
        #pragma unroll
        for (int bk = 0; bk < BlockTileK; bk += WarpTileK) {
            #pragma unroll
            for (int i = 0; i < frag_m_size; i++) {
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                    : "=r"(regA[i][0]), "=r"(regA[i][1])
                    : "l"(smem_warpA_ptr + (i * MMA_M + lane_id % 16) * ldm_blockA + bk)
                );
            }
            #pragma unroll
            for (int i = 0; i < frag_n_size; i++) {
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x1.trans.b16 {%0}, [%1];"
                    : "=r"(regB[i])
                    : "l"(smem_warpB_ptr + (lane_id % 8 + bk) * ldm_blockB + i * MMA_N)
                );
            }
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
                        :"r"(regA[i][0]), "r"(regA[i][1]),
                         "r"(regB[j]),
                         "r"(regD[i][j][0]), "r"(regD[i][j][1])
                    );
                }
            }
        }
        __syncthreads();
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




gemm::base::GemmOutput mma_ldmatrix_trans(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    // B.transpose();
    
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
        mma_ldmatrix_trans_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
