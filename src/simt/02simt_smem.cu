#include "gemm.cuh"
#include "utils/tensor.hpp"
using namespace gemm::base;

#define BlockTileM 256
#define BlockTileN 128
#define BlockTileK 16
#define ThreadTileM 16
#define ThreadTileN 8


// 6.767962 ms, M=N=2048, K=1024, device 2080Ti
__global__ void simt_smem_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    constexpr int float4_element_num = 8;
    constexpr int ldm_blockA = BlockTileK;
    constexpr int ldm_blockB = BlockTileN;
    constexpr int ldm_blockC = BlockTileN;
    constexpr int ldm_blockA_f4size = BlockTileK / float4_element_num;
    constexpr int ldm_blockB_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_blockC_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_regC = ThreadTileN;
    constexpr int ldm_regC_f4size = ThreadTileN / float4_element_num;
    int ldm_A = K;
    int ldm_B = N;
    int ldm_C = N;
    int ldm_A_f4size = K / float4_element_num;
    int ldm_B_f4size = N / float4_element_num;
    int ldm_C_f4size = N / float4_element_num;


    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_offsetx = blockIdx.x * BlockTileN;
    int block_offsety = blockIdx.y * BlockTileM;
    
    __shared__ float4 smem_A[BlockTileM * ldm_blockA_f4size];
    __shared__ float4 smem_B[BlockTileK * ldm_blockB_f4size];

    float4 reg_a[ThreadTileM/float4_element_num]; // 4
    float4 reg_b[ThreadTileN/float4_element_num]; // 4
    float4 reg_c[ThreadTileM * ThreadTileN/float4_element_num]{0}; // 16
    // reinterpret_cast has no runtime cost.
    #define gmem_blockA_hf_ptr   (A + (block_offsety) * K)
    #define gmem_blockB_hf_ptr   (B + block_offsetx)
    #define gmem_blockC_hf_ptr   (C + (block_offsety) * N + block_offsetx)
    #define gmem_blockA_f4_ptr   reinterpret_cast<const float4*>(A + (block_offsety) * K)
    #define gmem_blockB_f4_ptr   reinterpret_cast<const float4*>(B + block_offsetx)
    #define gmem_blockC_f4_ptr   reinterpret_cast<float4*>(C + (block_offsety) * N + block_offsetx)

    #define smem_blockA_hf_ptr   reinterpret_cast<half*>(smem_A)
    #define smem_blockB_hf_ptr   reinterpret_cast<half*>(smem_B)
    #define smem_blockA_f4_ptr   smem_A
    #define smem_blockB_f4_ptr   smem_B

    #define reg_a_hf_ptr    reinterpret_cast<half*>(reg_a)
    #define reg_b_hf_ptr    reinterpret_cast<half*>(reg_b)
    #define reg_c_hf_ptr    reinterpret_cast<half*>(reg_c)
    #define reg_a_f4_ptr    reg_a
    #define reg_b_f4_ptr    reg_b
    #define reg_c_f4_ptr    reg_c

    int offset_st_global_cx = threadIdx.x * ThreadTileN;
    int offset_st_global_cy = threadIdx.y * ThreadTileM;
    int offset_ld_reg1_a = offset_st_global_cy;
    int offset_ld_reg1_b = offset_st_global_cx;

    for (int k = 0; k < K; k += BlockTileK) {
        // from global load a,b
        #pragma unroll
        for (int i = tid; i < BlockTileM * ldm_blockA_f4size; i += blockDim.x * blockDim.y)
        {
            int offset_ld2s_global_ax = i % ldm_blockA_f4size;
            int offset_ld2s_global_ay = i / ldm_blockA_f4size;
            int offset_st_smem_ax = offset_ld2s_global_ax;
            int offset_st_smem_ay = offset_ld2s_global_ay;
            // if cc < 8.0, it must be stored in shared memory through register storage.
            float4 buffer = gmem_blockA_f4_ptr[k / float4_element_num + offset_ld2s_global_ay * ldm_A_f4size + offset_ld2s_global_ax];
            smem_blockA_f4_ptr[offset_st_smem_ay * ldm_blockA_f4size + offset_st_smem_ax] = buffer;
        }

        #pragma unroll
        for (int i = tid; i < BlockTileK * ldm_blockB_f4size; i += blockDim.x * blockDim.y)
        {
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;
            int offset_ld2s_global_by = i / ldm_blockB_f4size;
            int offset_st_smem_bx = offset_ld2s_global_bx;
            int offset_st_smem_by = offset_ld2s_global_by;
            // if cc < 8.0, it must be stored in shared memory through register storage.
            float4 buffer = gmem_blockB_f4_ptr[k * N / float4_element_num + offset_ld2s_global_by * ldm_B_f4size + offset_ld2s_global_bx];
            smem_blockB_f4_ptr[offset_st_smem_by * ldm_blockB_f4size + offset_st_smem_bx] = buffer;
        }
        // it's impotant to sync threads before using shared memory, make sure smem data is freeze when register is reading
        __syncthreads();

        #pragma unroll
        for (int bk = 0; bk < BlockTileK; bk++)
        {
            // ld reg_a element by element due to column is not continuous
            #pragma unroll
            for (int i = 0; i < ThreadTileM; i++)
            {
                reg_a_hf_ptr[i] = smem_blockA_hf_ptr[(offset_ld_reg1_a + i) * ldm_blockA + bk];
            }
            // ld reg_b float4
            #pragma unroll
            for (int i = 0; i < ThreadTileN / float4_element_num; i++)
            {
                reg_b_f4_ptr[i] = smem_blockB_f4_ptr[bk * ldm_blockB_f4size + offset_ld_reg1_b/float4_element_num + i];
            }

            // compute
            #pragma unroll
            for (int i = 0; i < ThreadTileM; i++)
            {
                #pragma unroll
                for (int j = 0; j < ThreadTileN; j++)
                {
                    reg_c_hf_ptr[i * ldm_regC + j] += reg_a_hf_ptr[i] * reg_b_hf_ptr[j];
                }
            }
        }
        // faster warp may start next loop to modify smem if without this fence, make sure smem data is freeze when register is reading
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < ThreadTileM; i++) {
        #pragma unroll
        for (int j = 0; j < ThreadTileN / float4_element_num; j++) {
            gmem_blockC_f4_ptr[(offset_st_global_cy + i) * ldm_C_f4size + offset_st_global_cx/float4_element_num + j] = reg_c_f4_ptr[i * ldm_regC_f4size + j];
        }
    }
};


gemm::base::GemmOutput simt_smem(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    assert(BlockTileM % ThreadTileM == 0 && BlockTileN % ThreadTileN == 0 && BlockTileK % 8 == 0);
    assert(ThreadTileM % 8 == 0 && ThreadTileN % 8 == 0);

    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block(BlockTileN / ThreadTileN, BlockTileM / ThreadTileM);
    utils::Timeit t;
    for (int i = 0; i < launch_times; i++) {
        t.start();
        simt_smem_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
