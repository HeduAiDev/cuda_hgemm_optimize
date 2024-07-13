#pragma once

#ifdef __CUDA__
#include <cuda_fp16.h>
#endif

namespace tui
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// runable
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace runable {
        /// @brief 
        /// @param ptr  a matrix pointer
        /// @param rows  your matrix row size
        /// @param cols  your matrix col size
        /// @param screen_size_x    print size in x axis 
        /// @param screen_size_y    print size in y axis 

        // dragable scroll bar
        //  ↓                       
        // ████████████████████████__________________________
        // .0│-2.│0.0│-4.│-3.│-5.│-4.│1.0│-1.│-5.│1.0│-4.│3 
        // ──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───
        // .0│3.0│-5.│3.0│2.0│2.0│3.0│-2.│3.0│-2.│2.0│-4.│-9
        // ──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───
        // 5.│4.0│4.0│1.0│3.0│-2.│-1.│3.0│-1.│4.0│4.0│-3.│0  █
        // ──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───█
        // .0│-5.│3.0│3.0│-5.│1.0│3.0│-4.│4.0│3.0│4.0│2.0│-1 █
        // ──┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───█ <-dragable scroll bar

        // print matrix in tui that interacts with the mouse
        void print_matrix(float* ptr, int rows, int cols, int screen_size_x = 50, int screen_size_y = 15);
        void print_matrix(double* ptr, int rows, int cols, int screen_size_x = 50, int screen_size_y = 15);
        void print_matrix(int* ptr, int rows, int cols, int screen_size_x = 50, int screen_size_y = 15);
        #ifdef __CUDA__
        void print_matrix(half* ptr, int rows, int cols, int screen_size_x = 50, int screen_size_y = 15);
        #endif

         // static view but you can setting the offset of x and y by specify a center point 
        void print_matrix_glance(float *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x = 50, int screen_size_y = 15);
        void print_matrix_glance(double *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x = 50, int screen_size_y = 15);
        void print_matrix_glance(int *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x = 50, int screen_size_y = 15);
        #ifdef __CUDA__
        void print_matrix_glance(half *ptr, int rows, int cols, int row_id, int col_id, int screen_size_x = 50, int screen_size_y = 15);
        #endif


        void diff(float *ptr_a, float *ptr_b, int rows, int cols, float accuracy = 1e-3);
        void diff(double *ptr_a, double *ptr_b, int rows, int cols, float accuracy = 1e-3);
        void diff(int *ptr_a, int *ptr_b, int rows, int cols, float accuracy = 1e-3);
        #ifdef __CUDA__
        void diff(half *ptr_a, half *ptr_b, int rows, int cols, float accuracy = 1e-3);
        #endif
    } // namespace runable
    
} // namespace tui
