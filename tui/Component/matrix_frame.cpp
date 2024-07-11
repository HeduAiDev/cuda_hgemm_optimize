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
        MatrixFrameBase<T>::MatrixFrameBase(MatrixFrameOptions<T>& options) : MatrixFrameOptions<T>(options) {
            col_labels_ = getColLabels();
            row_labels_ = getRowLabels();
            SliderOption<float> slider_x_option = {this -> focus_x, 0.0f, 1.0f, 0.01f, Direction::Right, Color::White, Color::Grey50};
            SliderOption<float> slider_y_option = {this -> focus_y, 0.0f, 1.0f, 0.01f, Direction::Down, Color::White, Color::Grey50};
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
                            {col_labels_ | focusPositionRelative(this -> focus_x(), 0) | frame | size(HEIGHT, EQUAL, 1)},
                            {matrix_ | focusPositionRelative(this -> focus_x(), this -> focus_y()) | frame,},
                        }),
                    }) | flex,
                    vbox({
                        text(" ") | size(HEIGHT, EQUAL, 2),
                        hbox({
                            row_labels_ | focusPositionRelative(0, this -> focus_y()) | frame,
                            slider_y_ -> Render()
                        }) | yflex
                    }) | size(WIDTH, EQUAL, 4)
                })
            });
        }
        
        template<typename T>
        Element MatrixFrameBase<T>::getColLabels() {
            ::std::vector<Element> col_labels_arr;
            for (int i = 0; i < this -> cols; i ++) {
                col_labels_arr.push_back(text(::std::to_string(i)) | center | frame | size(WIDTH, EQUAL, text_width_) | color(Color::Gold3Bis) | bgcolor(Color::Grey3));
                col_labels_arr.push_back(separator() | color(Color::Gold3) | bgcolor(Color::Grey3));
            }
            return gridbox({col_labels_arr});
        }

        template<typename T>
        Element MatrixFrameBase<T>::getRowLabels() {
            ::std::vector<::std::vector<Element>> row_labels_arr;
            for (int i = 0; i < this -> rows - 1; i ++) {
                row_labels_arr.push_back({text(::std::to_string(i)) | size(HEIGHT, EQUAL, 1) | center | color(Color::Gold3Bis) | bgcolor(Color::Grey3)});
                row_labels_arr.push_back({separator() | color(Color::Gold3) | bgcolor(Color::Grey3)});
            }
            row_labels_arr.push_back({text(::std::to_string(this -> rows - 1)) | size(HEIGHT, EQUAL, 1) | center | color(Color::Gold3Bis) | bgcolor(Color::Grey3)});
            return gridbox(row_labels_arr);
        }

        template <typename T>
        Element  MatrixFrameBase<T>::getMatrix() {
            ::std::vector<Elements> _rows_arr;
            for (int i = 0; i < this -> rows; i++) {
                ::std::vector<Element> _cols_arr;
                ::std::vector<Element> _separator_arr;
                for (int j = 0; j < this -> cols; j++) {
                    T val = this -> ptr[i * this -> cols + j];
                    // │ele│
                    // ┼───┼
                    Element ele = text(std::to_string(val)) | center | frame | size(WIDTH, EQUAL, 3) | size(HEIGHT, EQUAL, 1); 
                    // |
                    Element separator_right = separator();
                    // ───
                    Element separator_bottom = separator();
                    // ┼
                    Element separator_cross = separator();
                    if (this -> element_style != nullptr) {
                        this -> element_style(ele, j, i, separator_right, separator_bottom, separator_cross);
                    }
                    _cols_arr.push_back(ele);
                    _cols_arr.push_back(separator_right);
                    _separator_arr.push_back(separator_bottom);
                    _separator_arr.push_back(separator_cross);
                }
                _rows_arr.push_back(_cols_arr);
                if (i != this -> rows - 1) {
                    _rows_arr.push_back(_separator_arr);
                }
            }
            return gridbox(_rows_arr);
        }

        template<typename T>
        float& MatrixFrameBase<T>::getFocusX() {
            return this -> focus_x;
        }

        template<typename T>
        float& MatrixFrameBase<T>::getFocusY() {
            return this -> focus_y;
        }

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, Color color)> MatrixFrameOptionsCommonElementStyle::hight_light_row = [](int row_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (y == row_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_right |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int col_id, Color color)> MatrixFrameOptionsCommonElementStyle::hight_light_col = [](int col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x == col_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, int col_id, Color color)> MatrixFrameOptionsCommonElementStyle::mark_point = [](int row_id, int col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x == col_id && y == row_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id - 1 && y == row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id && y == row_id - 1) {
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id - 1 && y == row_id - 1) {
                    separator_cross |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, int col_id, Color trace_color, Color point_color)> MatrixFrameOptionsCommonElementStyle::mark_point_trace = [](int row_id, int col_id, Color trace_color, Color point_color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, col_id, trace_color, point_color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {

                mark_point(row_id, col_id, point_color)(ele, x, y, separator_right, separator_bottom, separator_cross);
                if ( x > col_id && y == row_id ) {
                    ele |= ::ftxui::bgcolor(trace_color);
                    separator_right |= ::ftxui::bgcolor(trace_color);
                }
                if ( x == col_id && y < row_id ) {
                    ele |= ::ftxui::bgcolor(trace_color);
                    separator_bottom |= ::ftxui::bgcolor(trace_color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle()> MatrixFrameOptionsCommonElementStyle::empty_style = []() -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {};
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color)> MatrixFrameOptionsCommonElementStyle::mark_sub_matrix = [](int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [left_up_row_id, left_up_col_id, right_bottom_row_id, right_bottom_col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x >= left_up_col_id && x <= right_bottom_col_id && y >= left_up_row_id && y <= right_bottom_row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                    ele |= ::ftxui::bgcolor(color);
                }
                // left border
                if (x == left_up_col_id - 1 && y >= left_up_row_id && y <= right_bottom_row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                // top border
                if (y == left_up_row_id - 1 && x >= left_up_col_id && x <= right_bottom_col_id) {
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                // left-top corss
                if (x == left_up_col_id - 1 && y == left_up_row_id - 1) {
                    separator_cross |= ::ftxui::bgcolor(color);
                }
            };
        };

        MatrixFrameOptionsCommonElementStyle::ElementStyle operator|(MatrixFrameOptionsCommonElementStyle::ElementStyle lhs, MatrixFrameOptionsCommonElementStyle::ElementStyle rhs) {
                return [lhs, rhs](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                    lhs(ele, x, y, separator_right, separator_bottom, separator_cross);
                    rhs(ele, x, y, separator_right, separator_bottom, separator_cross);
                };
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