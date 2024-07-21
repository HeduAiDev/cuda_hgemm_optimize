
#pragma once
#include <iostream>

#define TEST_IT(func, groundTruth, launch_cnt)                                            \
    {                                                                                     \
        {                                                                                 \
            auto res = func(A.hostPtr(), B.hostPtr(), C.hostPtr(), M, N, K, launch_cnt); \
            ::utils::test::print_centered(#func, 100, '=');                                               \
            printf("%s: %f ms\n", #func, res.excute_time_ms);                         \
            GroundTruth.checkResult(C);                                                   \
        }                                                                                 \
    }
namespace utils
{
    namespace test
    {
        template <typename T1, typename T2>
        __global__ void check_kernel(T1 *a, T2 *b, bool *flg, unsigned int n)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = tid; i < n; i += blockDim.x * gridDim.x)
            {
                if (abs((float)(a[i] - b[i]) > 1e-6))
                {
                    // printf("a[%d] = %.10f, b[%d] = %.10f\n", i, __half2float(a[i]), i, __half2float(b[i]));
                    *flg = false;
                }
                if (*flg == false)
                    return;
            }
        }

        template <int grid, int block, typename T1, typename T2>
        void check(T1 *h_a, T2 *h_b, unsigned int n, std::string suffix = "")
        {
            bool h_is_equal = true;
            T1 *d_a;
            T2 *d_b;
            bool *d_is_equal;
            CHECK(cudaMalloc((void **)&d_is_equal, sizeof(bool)));
            CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(bool), cudaMemcpyHostToDevice));
            cudaMalloc((void **)&d_a, n * sizeof(T1));
            cudaMalloc((void **)&d_b, n * sizeof(T2));
            CHECK(cudaMemcpy(d_a, h_a, n * sizeof(T1), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_b, h_b, n * sizeof(T2), cudaMemcpyHostToDevice));
            check_kernel<T1, T2><<<grid, block>>>(d_a, d_b, d_is_equal, n);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(bool), cudaMemcpyDeviceToHost));
            printf("%s is equal: %s\n", suffix.c_str(), h_is_equal == true ? "true" : "false");
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_is_equal);
        }

        template <class T>
        __host__ __device__  void print_matrix(T *a, int rows, int cols)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    printf("%5.0f", static_cast<float>(a[i * cols + j]));
                }
                printf("\n");
            }
            printf("\n");
        }

        void print_centered(const char *str, int width, char fill_char)
        {
            int len = strlen(str);
            if (len >= width)
            {
                // 如果字符串长度大于等于指定宽度，直接输出字符串
                printf("%s\n", str);
            }
            else
            {
                // 计算左右填充的长度
                int total_padding = width - len;
                int left_padding = total_padding / 2;
                int right_padding = total_padding - left_padding;

                // 输出左填充字符
                for (int i = 0; i < left_padding; i++)
                {
                    putchar(fill_char);
                }

                // 输出字符串
                printf("%s", str);

                // 输出右填充字符
                for (int i = 0; i < right_padding; i++)
                {
                    putchar(fill_char);
                }

                // 换行
                putchar('\n');
            }
        }

    }
}