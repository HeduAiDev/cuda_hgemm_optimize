#pragma once

#include <iostream>
#include <cstdlib>
#include <utils/base.hpp>
#include <cuda_runtime.h>
#include <assert.h>
#include <limits>

namespace utils
{
    namespace tensor
    {
        enum class StorageOrder
        {
            RowMajor,
            ColumnMajor
        };

        enum class InitializationType
        {
            Random,
            Zero
        };

        template <typename T>
        class Tensor
        {
        public:
            Tensor(int rows, int cols, StorageOrder order)
                : rows_(rows), cols_(cols), order_(order), hostData_(nullptr), deviceData_(nullptr)
            {
                hostData_ = new T[rows_ * cols_];
                CHECK(cudaMalloc((void **)&deviceData_, rows * cols * sizeof(T)));
            }

            Tensor(T* data, int rows, int cols, StorageOrder order)
                : rows_(rows), cols_(cols), order_(order), hostData_(data), deviceData_(nullptr)
            {
                own_host = false;
                CHECK(cudaMalloc((void **)&deviceData_, rows * cols * sizeof(T)));
            }

            ~Tensor()
            {
                if (own_host) {
                    delete[] hostData_;
                }
                if (own_device) {
                    cudaFree(deviceData_);
                }
            }

            void initializeHostData(InitializationType type)
            {
                for (int i = 0; i < rows_ * cols_; ++i)
                {
                    hostData_[i] = (type == InitializationType::Random) ? static_cast<T>(rand() % 10 - 5) : static_cast<T>(0);
                }
            }
            void copyToDevice()
            {
                CHECK(cudaMemcpy(deviceData_, hostData_, rows_ * cols_ * sizeof(T), cudaMemcpyHostToDevice));
            }

            void copyToHost()
            {
                CHECK(cudaMemcpy(hostData_, deviceData_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost));
            }

            int getRows() const {return rows_;};
            int getCols() const {return cols_;};
            T* hostPtr() const {return hostData_;};
            T* devicePtr() const {return deviceData_;};
            void printHostData(std::string label) const
            {
                std::cout << label << "Host Data:" << std::endl;
                // test::print_matrix(hostData_, rows_, cols_);
            }

            void checkResult(const Tensor<T>& ground_truth, float precision = 1e-3) const
            {
                assert(rows_ == ground_truth.getRows());
                assert(cols_ == ground_truth.getCols());
                float avg_diff = 0;
                float max_diff = std::numeric_limits<float>::lowest();
                int cnt = 0;
                for (int i = 0; i < rows_ * cols_; ++i)
                {
                    float diff = std::abs(static_cast<float>(hostData_[i] - ground_truth[i]));
                    max_diff = std::max(max_diff, diff);
                    avg_diff += diff;
                    cnt += (int)(diff > precision);
                }
                if (cnt == 0) {
                    avg_diff = 0;
                } else {
                    avg_diff /= cnt;
                }
                std::cout << "Is Equal: " << ((float)max_diff <= precision ? "true": "false") << std::endl;
                std::cout <<"Max diff: " << max_diff << ", Avg diff: " << avg_diff << std::endl;
            }

            T& operator[] (int idx) const {
                return hostData_[idx];
            }

            T& at(int row, int col) const
            {
                if (order_ == StorageOrder::RowMajor) {
                    return hostData_[row * cols_ + col];
                }
                else if (order_ == StorageOrder::ColumnMajor) {
                    return hostData_[col * rows_ + row];
                }
                    
            }

        private:
            int rows_;
            int cols_;
            StorageOrder order_;
            T *hostData_;
            T *deviceData_;
            bool own_host = true;
            bool own_device = true;
        };


    }
}


