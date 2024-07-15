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
        float elapsed_time;
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
        }
        float get_FLOPS(int m, int n, int k)
        {
            return ((float)m * n * k * 2) / (elapsed_time * 1e-3);
        }
    };

    struct __device_builtin__ __builtin_align__(16) float4
    {
        half data[8];
        __host__ __device__ half operator[](unsigned idx) const { return data[idx]; }
        __host__ __device__ half &operator[](unsigned idx) { return data[idx]; }
        __host__ __device__ float4 operator+(const float4 &rhs) {
            return {data[0] + rhs[0], data[1] + rhs[1], data[2] + rhs[2], data[3] + rhs[3],
            data[4] + rhs[4], data[5] + rhs[5], data[6] + rhs[6], data[7] + rhs[7]};
        }

        __host__ __device__ float4 operator*(const half &rhs) {
            return {data[0] * rhs, data[1] * rhs, data[2] * rhs, data[3] * rhs,
            data[4] * rhs, data[5] * rhs, data[6] * rhs, data[7] * rhs};
        }

    };
};