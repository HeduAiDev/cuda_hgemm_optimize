#include "gemm.cuh"
#include "utils/tensor.hpp"
using namespace gemm::base;

#define BlockTileM 256
#define BlockTileN 128
#define ThreadTileM 16
#define ThreadTileN 16

// 7.501184 ms, M=N=2048, K=1024, device 2080Ti
//Improve REGister Computational Intensity
__global__ void simt_regci_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    constexpr int float4_element_num = 8;
    constexpr int ldm_regC = ThreadTileN;
    constexpr int ldm_regC_f4size = ThreadTileN / float4_element_num;
    int ldm_A = K;
    int ldm_B = N;
    int ldm_C = N;
    int ldm_A_f4size = K / float4_element_num;
    int ldm_B_f4size = N / float4_element_num;
    int ldm_C_f4size = N / float4_element_num;

    int block_offsetx = blockIdx.x * BlockTileN;
    int block_offsety = blockIdx.y * BlockTileM;

    int offset_st_global_cx = threadIdx.x * ThreadTileN;
    int offset_st_global_cy = threadIdx.y * ThreadTileM;
    int offset_ld_reg1_a = offset_st_global_cy;
    int offset_ld_reg1_b = offset_st_global_cx;
    float4 reg_a[ThreadTileM/float4_element_num];
    float4 reg_b[ThreadTileN/float4_element_num];
    float4 reg_c[ThreadTileM * ThreadTileN/float4_element_num]{0};
    // reinterpret_cast has no runtime cost.
    #define gmem_blockA_hf_ptr   (A + (block_offsety) * K)
    #define gmem_blockB_hf_ptr   (B + block_offsetx)
    #define gmem_blockC_hf_ptr   (C + (block_offsety) * N + block_offsetx)
    #define gmem_blockA_f4_ptr   reinterpret_cast<const float4*>(A + (block_offsety) * K)
    #define gmem_blockB_f4_ptr   reinterpret_cast<const float4*>(B + block_offsetx)
    #define gmem_blockC_f4_ptr   reinterpret_cast<float4*>(C + (block_offsety) * N + block_offsetx)
    #define reg_a_hf_ptr    reinterpret_cast<half*>(reg_a)
    #define reg_b_hf_ptr    reinterpret_cast<half*>(reg_b)
    #define reg_c_hf_ptr    reinterpret_cast<half*>(reg_c)
    #define reg_a_f4_ptr    reg_a
    #define reg_b_f4_ptr    reg_b
    #define reg_c_f4_ptr    reg_c

    for (int k = 0; k < K; k++) {
        // ld reg_a element by element due to column is not continuous
        #pragma unroll
        for (int i = 0; i < ThreadTileM; i++) {
            reg_a_hf_ptr[i] = gmem_blockA_hf_ptr[(offset_ld_reg1_a + i) * ldm_A + k];
        }
        // ld reg_b float4
        #pragma unroll
        for (int i = 0; i < ThreadTileN / float4_element_num; i++) {
            reg_b_f4_ptr[i] = gmem_blockB_f4_ptr[k * ldm_B_f4size + offset_ld_reg1_b / float4_element_num + i];
        }

        // compute
        #pragma unroll
        for (int i = 0; i < ThreadTileM; i++) {
            #pragma unroll
            for (int j = 0; j < ThreadTileN; j++) {
                reg_c_hf_ptr[i * ldm_regC + j] += reg_a_hf_ptr[i] * reg_b_hf_ptr[j];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < ThreadTileM; i++) {
        #pragma unroll
        for (int j = 0; j < ThreadTileN / float4_element_num; j++) {
            gmem_blockC_f4_ptr[(offset_st_global_cy + i) * ldm_C_f4size + offset_st_global_cx / float4_element_num + j] = reg_c_f4_ptr[i * ldm_regC_f4size + j];
        }
    }
};



gemm::base::GemmOutput simt_regci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    assert(BlockTileM % ThreadTileM == 0 && BlockTileN % ThreadTileN == 0);

    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block(BlockTileN / ThreadTileN, BlockTileM / ThreadTileM);
    utils::Timeit t;
    for (int i = 0; i < launch_times; i++) {
        t.start();
        simt_regci_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
