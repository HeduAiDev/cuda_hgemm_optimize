#include "gemm.cuh"
#include "utils/tensor.hpp"
using namespace gemm::base;

#define BlockTileM 128
#define BlockTileN 128
#define BlockTileK 16
#define ThreadTileM 8
#define ThreadTileN 8
// reference: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/#software_pipelining
// ←-----------------------------------------------------------------------------------
// ⤷---------------------------------------iter k-----------------------------------→-⤴
// |████████████████load global███████████████████████|███store shared███|             |  Global to Shared Memory
// |---------------------------------------iter bk-----------------------↘-------------|
// |█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|  Shared Memory to Registers
// ↘-------------↘------------↘-------------↘-------------↘-------------↘-------------↘
// |████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|  Registers to CUDA cores

#define LOAD_GLOBAL(K)                                                                                                                                                 \
    {                                                                                                                                                                  \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * ldm_blockA_f4size; i += total_threads)                                                                    \
        {                                                                                                                                                              \
            int offset_ld2s_global_ax = i % ldm_blockA_f4size;                                                                                                         \
            int offset_ld2s_global_ay = i / ldm_blockA_f4size;                                                                                                         \
            buffer_a[i / total_threads] = gmem_blockA_f4_ptr[(K) / float4_element_num + offset_ld2s_global_ay * ldm_A_f4size + offset_ld2s_global_ax];                 \
        }                                                                                                                                                              \
        _Pragma("unroll") for (int i = tid; i < BlockTileK * ldm_blockB_f4size; i += total_threads)                                                                    \
        {                                                                                                                                                              \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                         \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                         \
            buffer_b[i / total_threads] = gmem_blockB_f4_ptr[(K) * N / float4_element_num + offset_ld2s_global_by * ldm_B_f4size + offset_ld2s_global_bx];             \
        }                                                                                                                                                              \
    }

#define STORE_SHARED(WRITE_IDX)                                                                                                                                                             \
    {                                                                                                                                                                                       \
        _Pragma("unroll") for (int i = tid; i < BlockTileM * ldm_blockA_f4size; i += total_threads)                                                                                         \
        {                                                                                                                                                                                   \
            int offset_ld2s_global_ax = i % ldm_blockA_f4size;                                                                                                                              \
            int offset_ld2s_global_ay = i / ldm_blockA_f4size;                                                                                                                              \
            int offset_st_smem_ax = offset_ld2s_global_ay;                                                                                                                                  \
            int offset_st_smem_ay = offset_ld2s_global_ax;                                                                                                                                  \
            for (int e = 0; e < float4_element_num; e++)                                                                                                                                    \
            {                                                                                                                                                                               \
                smem_blockA_hf_ptr((WRITE_IDX))[(offset_st_smem_ay * float4_element_num + e) * BlockTileM + offset_st_smem_ax] = reinterpret_cast<half *>(&buffer_a[i / total_threads])[e]; \
            }                                                                                                                                                                               \
        }                                                                                                                                                                                   \
        _Pragma("unroll") for (int i = tid; i < BlockTileK * ldm_blockB_f4size; i += total_threads)                                                                                         \
        {                                                                                                                                                                                   \
            int offset_ld2s_global_bx = i % ldm_blockB_f4size;                                                                                                                              \
            int offset_ld2s_global_by = i / ldm_blockB_f4size;                                                                                                                              \
            int offset_st_smem_bx = offset_ld2s_global_bx;                                                                                                                                  \
            int offset_st_smem_by = offset_ld2s_global_by;                                                                                                                                  \
            smem_blockB_f4_ptr((WRITE_IDX))[offset_st_smem_by * ldm_blockB_f4size + offset_st_smem_bx] = buffer_b[i / total_threads];                                                       \
        }                                                                                                                                                                                   \
    }

#define LOAD_SHARED(BK, SMEM_READ_IDX, REG_WRITE_IDX)                                                                                                                 \
    {                                                                                                                                                                 \
        _Pragma("unroll") for (int i = 0; i < ThreadTileM / float4_element_num; i++)                                                                                  \
        {                                                                                                                                                             \
            reg_a_f4_ptr((REG_WRITE_IDX))[i] = smem_blockA_f4_ptr(SMEM_READ_IDX)[(BK) * BlockTileM / float4_element_num + offset_ld_reg1_a / float4_element_num + i]; \
        }                                                                                                                                                             \
        _Pragma("unroll") for (int i = 0; i < ThreadTileN / float4_element_num; i++)                                                                                  \
        {                                                                                                                                                             \
            reg_b_f4_ptr((REG_WRITE_IDX))[i] = smem_blockB_f4_ptr(SMEM_READ_IDX)[(BK) * ldm_blockB_f4size + offset_ld_reg1_b / float4_element_num + i];               \
        }                                                                                                                                                             \
    }

// 6.647104 ms, M=N=2048, K=1024, device 2080Ti
__global__ void simt_pipline_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    constexpr int float4_element_num = 8;
    constexpr int ldm_blockA = BlockTileK;
    constexpr int ldm_blockB = BlockTileN;
    constexpr int ldm_blockC = BlockTileN;
    constexpr int ldm_blockA_f4size = BlockTileK / float4_element_num;
    constexpr int ldm_blockB_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_blockC_f4size = BlockTileN / float4_element_num;
    constexpr int ldm_regC = ThreadTileN;
    constexpr int ldm_regC_f4size = ThreadTileN / float4_element_num;
    constexpr int total_threads = (BlockTileM / ThreadTileN) * (BlockTileN / ThreadTileN);
    constexpr int buffer_a_size = divCeil(BlockTileM * ldm_blockA_f4size, total_threads);
    constexpr int buffer_b_size = divCeil(BlockTileK * ldm_blockB_f4size, total_threads);
    int ldm_A = K;
    int ldm_B = N;
    int ldm_C = N;
    int ldm_A_f4size = K / float4_element_num;
    int ldm_B_f4size = N / float4_element_num;
    int ldm_C_f4size = N / float4_element_num;


    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_offsetx = blockIdx.x * BlockTileN;
    int block_offsety = blockIdx.y * BlockTileM;
    
    __shared__ float4 smem_A[2][BlockTileM * ldm_blockA_f4size];
    __shared__ float4 smem_B[2][BlockTileK * ldm_blockB_f4size];

    float4 reg_a[2][ThreadTileM/float4_element_num];
    float4 reg_b[2][ThreadTileN/float4_element_num];
    float4 reg_c[ThreadTileM * ThreadTileN/float4_element_num]{0};

    float4 buffer_a[buffer_a_size];
    float4 buffer_b[buffer_b_size];
    // reinterpret_cast has no runtime cost.
    #define gmem_blockA_hf_ptr   (A + (block_offsety) * K)
    #define gmem_blockB_hf_ptr   (B + block_offsetx)
    #define gmem_blockC_hf_ptr   (C + (block_offsety) * N + block_offsetx)
    #define gmem_blockA_f4_ptr   reinterpret_cast<const float4*>(A + (block_offsety) * K)
    #define gmem_blockB_f4_ptr   reinterpret_cast<const float4*>(B + block_offsetx)
    #define gmem_blockC_f4_ptr   reinterpret_cast<float4*>(C + (block_offsety) * N + block_offsetx)

    #define smem_blockA_hf_ptr(idx)   reinterpret_cast<half*>(smem_A[(idx)])
    #define smem_blockB_hf_ptr(idx)   reinterpret_cast<half*>(smem_B[(idx)])
    #define smem_blockA_f4_ptr(idx)   smem_A[(idx)]
    #define smem_blockB_f4_ptr(idx)   smem_B[(idx)]

    #define reg_a_hf_ptr(idx)    reinterpret_cast<half*>(reg_a[(idx)])
    #define reg_b_hf_ptr(idx)    reinterpret_cast<half*>(reg_b[(idx)])
    #define reg_a_f4_ptr(idx)    reg_a[(idx)]
    #define reg_b_f4_ptr(idx)    reg_b[(idx)]
    #define reg_c_hf_ptr    reinterpret_cast<half*>(reg_c)
    #define reg_c_f4_ptr    reg_c

    int offset_st_global_cx = threadIdx.x * ThreadTileN;
    int offset_st_global_cy = threadIdx.y * ThreadTileM;
    int offset_ld_reg1_a = offset_st_global_cy;
    int offset_ld_reg1_b = offset_st_global_cx;
    bool smem_write_idx = 0;
    bool reg_write_idx = 0;
    // Global to Shared Memory
    LOAD_GLOBAL(0)
    STORE_SHARED(smem_write_idx)
    __syncthreads();
    // Shared Memory to Registers
    // load bk frist data, after sync, smem_read_idx equal smem_write_idx since smem finish load, and data is freezed
    LOAD_SHARED(0, smem_write_idx, reg_write_idx)
    // this loop we compute [0:-1] gmem blocks and load [1:] gmem blocks
    for (int k = 1; k < K / BlockTileK; k ++) {
        smem_write_idx = !smem_write_idx;
        LOAD_GLOBAL(k * BlockTileK)
        // we expect bk's iter number BlockTileK is always even, that's why reg_write_idx can hard code
        reg_write_idx = 1;
        #pragma unroll
        for (int bk = 0; bk < BlockTileK - 1; bk++)
        {
            // bk will participate in compute, we load bk + 1 data
            LOAD_SHARED(bk + 1, !(smem_write_idx), reg_write_idx)
            // compute
            #pragma unroll
            for (int i = 0; i < ThreadTileM; i++)
            {
                #pragma unroll
                for (int j = 0; j < ThreadTileN; j++)
                {
                    reg_c_hf_ptr[i * ldm_regC + j] += reg_a_hf_ptr(!reg_write_idx)[i] * reg_b_hf_ptr(!reg_write_idx)[j];
                }
            }
            reg_write_idx = !reg_write_idx;
        }
        // __syncthreads(); // this sync is not necessary, since above compute use another smem write idx
        STORE_SHARED(smem_write_idx)
        __syncthreads();
        // next bk frist data
        LOAD_SHARED(0, smem_write_idx, reg_write_idx)
        // compute
        #pragma unroll
        for (int i = 0; i < ThreadTileM; i++)
        {
            #pragma unroll
            for (int j = 0; j < ThreadTileN; j++)
            {
                reg_c_hf_ptr[i * ldm_regC + j] += reg_a_hf_ptr(!reg_write_idx)[i] * reg_b_hf_ptr(!reg_write_idx)[j];
            }
        }
    }
    // compute last gmem block, different from above, we don't need to load next block data
    smem_write_idx = !smem_write_idx;
    // we expect bk's iter number BlockTileK is always even, that's why reg_write_idx can hard code
    reg_write_idx = 1;
    #pragma unroll
    for (int bk = 0; bk < BlockTileK; bk++)
    {
        // bk will participate in compute, we load bk + 1 data
        if (bk < BlockTileK -1) {
            LOAD_SHARED(bk + 1, !(smem_write_idx), reg_write_idx)
        }
        // compute
        #pragma unroll
        for (int i = 0; i < ThreadTileM; i++)
        {
            #pragma unroll
            for (int j = 0; j < ThreadTileN; j++)
            {
                reg_c_hf_ptr[i * ldm_regC + j] += reg_a_hf_ptr(!reg_write_idx)[i] * reg_b_hf_ptr(!reg_write_idx)[j];
            }
        }
        reg_write_idx = !reg_write_idx;
    }
    // store c
    #pragma unroll
    for (int i = 0; i < ThreadTileM; i++) {
        #pragma unroll
        for (int j = 0; j < ThreadTileN / float4_element_num; j++) {
            gmem_blockC_f4_ptr[(offset_st_global_cy + i) * ldm_C_f4size + offset_st_global_cx/float4_element_num + j] = reg_c_f4_ptr[i * ldm_regC_f4size + j];
        }
    }
};



gemm::base::GemmOutput simt_pipline(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, const int launch_times) {
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
        simt_pipline_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
        t.stop();
        C.initializeHostData(InitializationType::Zero);
        C.copyToHost();
    }
    return gemm::base::GemmOutput(cudaGetLastError(), t.cumulative_time / launch_times);
};
