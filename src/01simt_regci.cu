#include "gemm.cuh"
#include "utils/tensor.hpp"
using namespace gemm::base;

#define BlockTileM 32
#define BlockTileN 32
#define ThreadTileM 8
#define ThreadTileN 8

//Improve REGister Computational Intensity
__global__ void simt_regci_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {

    int offset_st_global_cx = blockIdx.x * BlockTileN + threadIdx.x * ThreadTileN;
    int offset_st_global_cy = blockIdx.y * BlockTileM + threadIdx.y * ThreadTileM;
    int offset_ld_reg1_a = offset_st_global_cy;
    int offset_ld_reg1_b = offset_st_global_cx;
    constexpr int float4_element_num = 8;
    utils::float4 reg_a[ThreadTileM/float4_element_num];
    utils::float4 reg_b[ThreadTileN/float4_element_num];
    utils::float4 reg_c[ThreadTileM][ThreadTileN/float4_element_num];
    for (int k = 0; k < K; k++) {
        // ld reg_a element by element due to column is not continuous
        #pragma unroll
        for (int i = 0; i < ThreadTileM / float4_element_num; i++) {
            #pragma unroll
            for (int j = 0; j < float4_element_num; j++) {
                reg_a[i][j] = A[(offset_ld_reg1_a + i * float4_element_num + j) * K + k];
            }
        }
        // ld reg_b float4
        #pragma unroll
        for (int i = 0; i < ThreadTileN / float4_element_num; i++) {
            reg_b[i] = reinterpret_cast<utils::float4 *>(&B[k * N + (offset_ld_reg1_b + i * float4_element_num)])[0];
        }

        // compute
        #pragma unroll
        for (int i = 0; i < ThreadTileM / float4_element_num; i++) {
            #pragma unroll
            for (int j = 0; j < ThreadTileN / float4_element_num; j++) {
                #pragma unroll
                for (int e = 0; e < float4_element_num; e++) {
                    reg_c[i * float4_element_num + e][j] = reg_b[j] * reg_a[i][e] + reg_c[i * float4_element_num + e][j];
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < ThreadTileM; i++) {
        #pragma unroll
        for (int j = 0; j < ThreadTileN / float4_element_num; j++) {
            reinterpret_cast<utils::float4 *>(&C[(offset_st_global_cy + i) * N + offset_st_global_cx + j * float4_element_num])[0] = reg_c[i][j];
        }
    }
};



gemm::base::GemmOutput simt_regci(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K) {
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
    t.start();
    simt_regci_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
    t.stop();
    C.copyToHost();
    return gemm::base::GemmOutput(cudaGetLastError(), t.elapsed_time);
};
