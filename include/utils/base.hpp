#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#define divCeil(a, b) (((a) + (b) - 1) / (b))

#define CHECK(call)                                                                   \
    {                                                                                 \
        const cudaError_t error = call;                                               \
        if (error != cudaSuccess)                                                     \
        {                                                                             \
            fprintf(stderr, "ERROR: %s:%d,", __FILE__, __LINE__);                     \
            fprintf(stderr, "code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            throw std::runtime_error("cuda error");                                       \
        }                                                                             \
    }


namespace utils
{


    struct Timeit
    {
        cudaEvent_t e_start;
        cudaEvent_t e_stop;
        float elapsed_time = 0;
        float cumulative_time = 0;
        void start()
        {
            CHECK(cudaEventCreate(&e_start));
            CHECK(cudaEventCreate(&e_stop));
            CHECK(cudaEventRecord(e_start, 0));
        }
        void stop()
        {
            CHECK(cudaEventRecord(e_stop, 0));
            CHECK(cudaEventSynchronize(e_stop));
            CHECK(cudaEventElapsedTime(&elapsed_time, e_start, e_stop));
            CHECK(cudaEventDestroy(e_start));
            CHECK(cudaEventDestroy(e_stop));
            cumulative_time += elapsed_time;
        }
        float get_FLOPS(int m, int n, int k)
        {
            return ((float)m * n * k * 2) / (elapsed_time * 1e-3);
        }
    };

    struct __device_builtin__ __builtin_align__(16) half8
    {
        float4 data{0};
        __host__ __device__ half8() {
        }
        __host__ __device__ half8(const half8 &other) {
            data = other.data;
        };
        __host__ __device__ half8(const half half1, const half half2, const half half3, const half half4,
        const half half5, const half half6, const half half7, const half half8) {
            half* ptr = (half*)&data;
            ptr[0] = half1;
            ptr[1] = half2;
            ptr[2] = half3;
            ptr[3] = half4;
            ptr[4] = half5;
            ptr[5] = half6;
            ptr[6] = half7;
            ptr[7] = half8;
        };
        __host__ __device__ half operator[](unsigned idx) const { return ((half*)&data)[idx]; }
        __host__ __device__ half &operator[](unsigned idx) { return ((half*)&data)[idx]; }
        __host__ __device__ half8 operator+(const half8 &rhs) {
            half* ptr = (half*)&data;
            return {ptr[0] + rhs[0], ptr[1] + rhs[1], ptr[2] + rhs[2], ptr[3] + rhs[3],
            ptr[4] + rhs[4], ptr[5] + rhs[5], ptr[6] + rhs[6], ptr[7] + rhs[7]};
        }

        __host__ __device__ half8 operator*(const half &rhs) {
            half* ptr = (half*)&data;
            return {ptr[0] * rhs, ptr[1] * rhs, ptr[2] * rhs, ptr[3] * rhs,
            ptr[4] * rhs, ptr[5] * rhs, ptr[6] * rhs, ptr[7] * rhs};
        }

    };
};