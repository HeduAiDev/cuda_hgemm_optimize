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

};