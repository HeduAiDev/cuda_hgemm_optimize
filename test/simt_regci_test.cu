#include "gemm.cuh"
#include "utils/tensor.hpp"
#include "utils/base.hpp"
#include "tui_tool_sets_runable.hpp"
using namespace utils;


int main() {
    int M = 128;
    int N = 128;
    int K = 128;
    auto A = tensor::Tensor<half>(M, K, tensor::StorageOrder::RowMajor);
    A.initializeHostData(tensor::InitializationType::Random);
    auto B = tensor::Tensor<half>(K, N, tensor::StorageOrder::RowMajor);
    B.initializeHostData(tensor::InitializationType::Random);
    auto C = tensor::Tensor<half>(M, N, tensor::StorageOrder::RowMajor);
    C.initializeHostData(tensor::InitializationType::Zero);
    auto GroundTruth = tensor::Tensor<half>(M, N, tensor::StorageOrder::RowMajor);
    GroundTruth.initializeHostData(tensor::InitializationType::Zero);
    simt_regci(A.hostPtr(), B.hostPtr(), C.hostPtr(), M, N, K);
    cuBLASGemm(A.hostPtr(), B.hostPtr(), GroundTruth.hostPtr(), M, N, K);
    ::tui::runable::diff(C.hostPtr(), GroundTruth.hostPtr(), M, N);
    // ::tui::runable::print_matrix(C.hostPtr(), M, N);
    return 0;
}