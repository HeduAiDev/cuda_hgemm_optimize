#include "gemm.hpp"
#include "utils/tensor.hpp"
using namespace gemm::base;




__global__ void simt_naive_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int offset_st_global_cx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset_st_global_cy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_ld_reg1_a = offset_st_global_cy;
    int offset_ld_reg1_b = offset_st_global_cx;
    half sum = 0;
    for (int k = 0; k < K; k++) {
        half reg1_a = A[offset_ld_reg1_a * K + k];
        half reg1_b = B[k * N + offset_ld_reg1_b];
        sum += reg1_a * reg1_b;
    }
    C[offset_st_global_cy * N + offset_st_global_cx] = sum;
};




gemm::base::GemmOutput simt_naive(half* A_ptr, half *B_ptr, half *C_ptr, int M, int N, int K, SimtNaiveOptions options) {
    using namespace utils::tensor;
    Tensor<half> A = Tensor<half>( A_ptr, M, K, StorageOrder::RowMajor );
    Tensor<half> B = Tensor<half>( B_ptr, K, N, StorageOrder::RowMajor );
    Tensor<half> C = Tensor<half>( C_ptr, M, N, StorageOrder::RowMajor );
    
    A.copyToDevice();
    B.copyToDevice();
    C.copyToDevice();

    int BlockTileM = options.get("BlockTileM");
    int BlockTileN = options.get("BlockTileN");
    dim3 grid(divCeil(N, BlockTileN), divCeil(M, BlockTileM));
    dim3 block(BlockTileN, BlockTileM);
    utils::Timeit t;
    t.start();
    simt_naive_kernel<<<grid, block>>>(A.devicePtr(), B.devicePtr(), C.devicePtr(), M, N, K);
    t.stop();
    C.copyToHost();
    return gemm::base::GemmOutput(cudaGetLastError(), t.elapsed_time);
};
