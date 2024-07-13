#pragma once

#include <iostream>
#include <cstdlib>
#include <utils/base.hpp>
#include <utils/test.hpp>
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
            Tensor(int rows, int cols, StorageOrder order = StorageOrder::RowMajor);
            ~Tensor();
            void initializeHostData(InitializationType type = InitializationType::Zero);
            void copyToDevice();
            void copyToHost();
            int getRows() const {return rows_;};
            int getCols() const {return cols_;};
            T* hostPtr() const {return hostData_;};
            T* devicePtr() const {return deviceData_;};
            void printHostData(std::string label = "") const;
            void checkResult(const Tensor<T>& ground_truth, float precision = 1e-3f) const;
            T& operator[](int idx) const;
            T& at(int row, int col) const;

        private:
            int rows_;
            int cols_;
            StorageOrder order_;
            T *hostData_;
            T *deviceData_;
        };

        template <typename T>
        Tensor<T>::Tensor(int rows, int cols, StorageOrder order)
            : rows_(rows), cols_(cols), order_(order), hostData_(nullptr), deviceData_(nullptr)
        {
            hostData_ = new T[rows_ * cols_];
            CHECK(cudaMalloc(&deviceData_, rows * cols * sizeof(T)));
        }

        template <typename T>
        Tensor<T>::~Tensor()
        {
            delete[] hostData_;
            cudaFree(deviceData_);
        }

        template <typename T>
        void Tensor<T>::initializeHostData(InitializationType type)
        {
            for (int i = 0; i < rows_ * cols_; ++i)
            {
                hostData_[i] = (type == InitializationType::Random) ? static_cast<T>(rand() % 10 - 5) : static_cast<T>(0);
            }
        }

        template <typename T>
        void Tensor<T>::copyToDevice()
        {
            cudaMemcpy(deviceData_, hostData_, rows_ * cols_ * sizeof(T), cudaMemcpyHostToDevice);
        }

        template <typename T>
        void Tensor<T>::copyToHost()
        {
            cudaMemcpy(hostData_, deviceData_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost);
        }

        template <typename T>
        void Tensor<T>::printHostData(std::string label) const
        {
            std::cout << label << "Host Data:" << std::endl;
            test::print_matrix(hostData_, rows_, cols_);
        }

        template <typename T>
        void Tensor<T>::checkResult(const Tensor<T>& ground_truth, float precision) const
        {
            assert(rows_ == ground_truth.getRows());
            assert(cols_ == ground_truth.getCols());
            float avg_diff = 0;
            float max_diff = std::numeric_limits<float>::lowest();
            for (int i = 0; i < rows_ * cols_; ++i)
            {
                float diff = std::abs(static_cast<float>(hostData_[i] - ground_truth[i]));
                max_diff = std::max(max_diff, diff);
                avg_diff += diff;
            }
            avg_diff /= rows_ * cols_;
            std::cout << "Is Equal: " << ((float)max_diff <= precision ? "${green}true${default}": "${red}false${default}") << std::endl;
            std::cout <<"Max diff: " << max_diff << ", Avg diff: " << avg_diff << std::endl;
        }


        template <typename T>
        T& Tensor<T>::operator[] (int idx) const {
            return hostData_[idx];
        }

        template <typename T>
        T &Tensor<T>::at(int row, int col) const
        {
            if (order_ == StorageOrder::RowMajor) {
                return hostData_[row * cols_ + col];
            }
            else if (order_ == StorageOrder::ColumnMajor) {
                return hostData_[col * rows_ + row];
            }
                
        }
    }
}


