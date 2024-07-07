#include "simt_naive.hpp"
#include "utils/test.hpp"
#include <utils/tensor.hpp>
#include <cuda_fp16.h>
using namespace utils;
int main() {
   auto t1 = tensor::Tensor<half>(3,4, tensor::StorageOrder::RowMajor);
   t1.initializeHostData(tensor::InitializationType::Random);
   t1.copyToDevice();
   t1.printHostData();
   
   auto t2 = tensor::Tensor<half>(3,4, tensor::StorageOrder::RowMajor);
   t2.initializeHostData(tensor::InitializationType::Random);
   t2.copyToDevice();
   t2.printHostData();
   
   t1.checkResult(t2);
}