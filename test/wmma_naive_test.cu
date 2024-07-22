#include "gemm.cuh"
#include "utils/tensor.hpp"
#include "utils/base.hpp"
#include "tui_tool_sets_runable.hpp"
#include "utils/test.hpp"
using namespace utils;


int main() {
    int M = 2048;
    int N = 2048;
    int K = 1024;
    srand(111);
    auto A = tensor::Tensor<half>(M, K, tensor::StorageOrder::RowMajor);
    A.initializeHostData(tensor::InitializationType::Random);
    auto B = tensor::Tensor<half>(K, N, tensor::StorageOrder::RowMajor);
    B.initializeHostData(tensor::InitializationType::Random);
    auto C = tensor::Tensor<half>(M, N, tensor::StorageOrder::RowMajor);
    C.initializeHostData(tensor::InitializationType::Zero);
    auto GroundTruth = tensor::Tensor<half>(M, N, tensor::StorageOrder::RowMajor);
    GroundTruth.initializeHostData(tensor::InitializationType::Zero);
    cuBLASGemm(A.hostPtr(), B.hostPtr(), GroundTruth.hostPtr(), M, N, K);
    TEST_IT(wmma_naive, GroundTruth, 100);
    // ::tui::runable::diff(C.hostPtr(), GroundTruth.hostPtr(), M, N);
    // ::tui::runable::print_matrix(A.hostPtr(), M, K);
    return 0;
}