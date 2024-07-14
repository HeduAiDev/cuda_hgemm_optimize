#include "utils/test.hpp"
#include <utils/tensor.hpp>
#include <cuda_fp16.h>
#include <tui_tool_sets_runable.hpp>
using namespace utils;
int main() {
   auto t1 = tensor::Tensor<half>(20,100, tensor::StorageOrder::RowMajor);
   t1.initializeHostData(tensor::InitializationType::Random);
   t1.copyToDevice();
   // t1.printHostData();

   // tui::runable::print_matrix(t1.hostPtr(), t1.getRows(), t1.getCols());
   // tui::runable::print_matrix_glance(t1.hostPtr(), t1.getRows(), t1.getCols(), 17, 10);
   
   auto t2 = tensor::Tensor<half>(20,100, tensor::StorageOrder::RowMajor);
   t2.initializeHostData(tensor::InitializationType::Random);
   t2.copyToDevice();
   // t2.printHostData();
   // tui::runable::print_matrix(t2.hostPtr(), t1.getRows(), t1.getCols());

   tui::runable::diff(t1.hostPtr(), t2.hostPtr(), t1.getRows(), t1.getCols());  // t1.checkResult(t2);
}