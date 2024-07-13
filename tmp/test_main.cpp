#include <stdio.h>
#include "include/tui_tool_sets_runable.hpp"
// #pragma comment(lib, "./lib/tui_tool_sets.lib")

int main() {
    int rows = 50;
    int cols = 50;
    float* a = new float[rows * cols];
    float* b = new float[rows * cols];
    tui::runable::diff(a, b, rows, cols);
    
    return 0;
}
