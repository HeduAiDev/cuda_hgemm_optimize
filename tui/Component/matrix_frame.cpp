#include <stddef.h>    // for size_t
#include <array>       // for array
#include <atomic>      // for atomic
#include <chrono>      // for operator""s, chrono_literals
#include <cmath>       // for sin
#include <functional>  // for ref, reference_wrapper, function
#include <memory>      // for allocator, shared_ptr, __shared_ptr_access
#include <string>  // for string, basic_string, char_traits, operator+, to_string
#include <thread>   // for sleep_for, thread
#include <utility>  // for move
#include <vector>   // for vector

#include "ftxui/component/component.hpp"  // for Checkbox, Renderer, Horizontal, Vertical, Input, Menu, Radiobox, ResizableSplitLeft, Tab
#include "ftxui/component/component_base.hpp"  // for ComponentBase, Component
#include "ftxui/component/component_options.hpp"  // for MenuOption, InputOption
#include "ftxui/component/event.hpp"              // for Event, Event::Custom
#include "ftxui/component/screen_interactive.hpp"  // for Component, ScreenInteractive
#include "ftxui/dom/elements.hpp"  // for text, color, operator|, bgcolor, filler, Element, vbox, size, hbox, separator, flex, window, graph, EQUAL, paragraph, WIDTH, hcenter, Elements, bold, vscroll_indicator, HEIGHT, flexbox, hflow, border, frame, flex_grow, gauge, paragraphAlignCenter, paragraphAlignJustify, paragraphAlignLeft, paragraphAlignRight, dim, spinner, LESS_THAN, center, yframe, GREATER_THAN
#include "ftxui/dom/flexbox_config.hpp"  // for FlexboxConfig
#include "ftxui/screen/color.hpp"  // for Color, Color::BlueLight, Color::RedLight, Color::Black, Color::Blue, Color::Cyan, Color::CyanLight, Color::GrayDark, Color::GrayLight, Color::Green, Color::GreenLight, Color::Magenta, Color::MagentaLight, Color::Red, Color::White, Color::Yellow, Color::YellowLight, Color::Default, Color::Palette256, ftxui
#include "ftxui/screen/color_info.hpp"  // for ColorInfo
#include "ftxui/screen/terminal.hpp"    // for Size, Dimensions
#include "tui_tool_sets.hpp"
namespace tui {
    namespace component {
        using namespace ftxui;
        
        template<typename T>
        MatrixFrameBase<T>::MatrixFrameBase(MatrixFrameOptions<T>& options) : MatrixFrameOptions(options) {
            col_labels_ = getColLabels();
            row_labels_ = getRowLabels();
            SliderOption<float> slider_x_option = {&focus_x, 0.0f, 1.0f, 0.01f, Direction::Right, Color::White, Color::Grey50};
            SliderOption<float> slider_y_option = {&focus_y, 0.0f, 1.0f, 0.01f, Direction::Down, Color::White, Color::Grey50};
            slider_x_ = Slider(slider_x_option) | bgcolor(Color::Grey23);
            slider_y_ = Slider(slider_y_option) | bgcolor(Color::Grey23);
            matrix_ = getMatrix();
            Add(Container::Vertical({
                slider_x_,
                slider_y_,
            }));
        }


        template<typename T>
        Element MatrixFrameBase<T>::Render() {
            return vbox({
                hbox({
                    vbox({
                        slider_x_ -> Render() | size(HEIGHT, EQUAL, 1),
                        gridbox({
                            {col_labels_ | focusPositionRelative(focus_x, 0) | frame | size(HEIGHT, EQUAL, 1)},
                            {matrix_ | focusPositionRelative(focus_x, focus_y) | frame,},
                        }),
                    }) | flex,
                    vbox({
                        text(" ") | size(HEIGHT, EQUAL, 2),
                        hbox({
                            row_labels_ | focusPositionRelative(0, focus_y) | frame,
                            slider_y_ -> Render()
                        }) | yflex
                    }) | size(WIDTH, EQUAL, 4)
                })
            })
        }
        
        template<typename T>
        Element MatrixFrameBase<T>::getColLabels() {
            ::std::vector<Element> col_labels_arr;
            for (int i = 0; i < cols; i ++) {
                col_labels_arr.push_back(text(::std::to_string(i)) | center | frame | size(WIDTH, EQUAL, text_width_) | color(Color::Gold3Bis) | bgcolor(Color::Grey3));
                col_labels_arr.push_back(separator() | color(Color::Gold3) | bgcolor(Color::Grey3));
            }
            return gridbox({col_labels_arr});
        }

        template<typename T>
        Element MatrixFrameBase<T>::getRowLabels() {
            ::std::vector<::std::vector<Element>> row_labels_arr;
            for (int i = 0; i < rows - 1; i ++) {
                row_labels_arr.push_back({text(::std::to_string(i)) | size(HEIGHT, EQUAL, 1) | center | color(Color::Gold3Bis) | bgcolor(Color::Grey3)});
                row_labels_arr.push_back({separator() | color(Color::Gold3) | bgcolor(Color::Grey3)});
            }
            row_labels_arr.push_back({text(::std::to_string(rows - 1)) | size(HEIGHT, EQUAL, 1) | center | color(Color::Gold3Bis) | bgcolor(Color::Grey3)});
            return gridbox(row_labels_arr);
        }

        template <typename T>
        Element  MatrixFrameBase<T>::getMatrix() {
            ::std::vector<Elements> _rows_arr;
            for (int i = 0; i < rows; i++) {
                ::std::vector<Element> _cols_arr;
                ::std::vector<Element> _separator_arr;
                for (int j = 0; j < cols; j++) {
                    T val = ptr[i * cols + j];
                    _cols_arr.push_back(text(std::to_string(val)) | center | frame | size(WIDTH, EQUAL, 3) | size(HEIGHT, EQUAL, 1));
                    _cols_arr.push_back(separator());
                    _separator_arr.push_back(separator());
                    _separator_arr.push_back(separator());
                }
                _rows_arr.push_back(_cols_arr);
                if (i != rows - 1) {
                    _rows_arr.push_back(_separator_arr);
                }
            }
            return gridbox(_rows_arr);
        }


        Component MatrixFrame(float* ptr, int rows, int cols, MatrixFrameOptions<float> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
            return Make<MatrixFrameBase<float>>(options);
        }
        Component MatrixFrame(int* ptr, int rows, int cols, MatrixFrameOptions<int> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
            return Make<MatrixFrameBase<int>>(options);
        }
        #ifdef __CUDA__
        Component MatrixFrame(half* ptr, int rows, int cols, MatrixFrameOptions<half> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
            return Make<MatrixFrameBase<half>>(options);
        }
        #endif
    }
}