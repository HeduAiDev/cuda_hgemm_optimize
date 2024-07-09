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
#include "tui_component.hpp"

namespace tui {
    namespace component {
        using namespace ftxui;
        Resizable4Blockbase::Resizable4Blockbase(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen,const Resizable4BlockOptions options) 
        : screen_(screen), options_(options) {
            block1_ = std::move(block1);
            block2_ = std::move(block2);
            block3_ = std::move(block3);
            block4_ = std::move(block4);

            Add(Container::Vertical({
                Container::Horizontal({
                    block1_,
                    block2_,
                }),
                Container::Horizontal({
                    block3_,
                    block4_,
                })
            }));
        };

        Element Resizable4Blockbase::Render()  {
            int block_width = screen_.dimx() / 2 + bias_x_;
            int block_height = screen_.dimy() / 2 + bias_y_;
            return vbox({
                hbox({
                    (options_.placeholder_block1 && isDragging() ? options_.placeholder_block1 : block1_ -> Render()) | size(WIDTH, EQUAL, block_width),
                    getVSeparator() | reflect(vseparator_up_box_),
                    (options_.placeholder_block2 && isDragging() ? options_.placeholder_block2 : block2_ -> Render()) | flex
                }) | size(HEIGHT, EQUAL, block_height),
                getHSeparator() | reflect(hseparator_box_),
                hbox({
                    (options_.placeholder_block3 && isDragging() ? options_.placeholder_block3 : block3_ -> Render()) | size(WIDTH, EQUAL, block_width),
                    getVSeparator() | reflect(vseparator_down_box_),
                    (options_.placeholder_block4 && isDragging() ? options_.placeholder_block4 : block4_ -> Render()) | flex
                }) | size(HEIGHT, EQUAL, screen_.dimy() - block_height)
            });
        };

        bool Resizable4Blockbase::OnEvent(Event event) {
            if (event.is_mouse())
            {
                OnMouseEvent(std::move(event));
            }
            return ComponentBase::OnEvent(std::move(event));
        };

        bool Resizable4Blockbase::OnMouseEvent(Event event) {
            is_hover_hseparator_ = hseparator_box_.Contain(event.mouse().x, event.mouse().y);
            is_hover_vseparator_up_ = vseparator_up_box_.Contain(event.mouse().x, event.mouse().y);
            is_hover_vseparator_down_ = vseparator_down_box_.Contain(event.mouse().x, event.mouse().y);

            if (isDragging() && event.mouse().motion == Mouse::Released) {
                is_dragging_hseparator_ = false;
                is_dragging_vseparator_up_ = false;
                is_dragging_vseparator_down_ = false;
                return false;
            }
            if (event.mouse().button == Mouse::Left && event.mouse().motion == Mouse::Pressed && !isDragging()) {
                if (is_hover_hseparator_) {
                    is_dragging_hseparator_ = true;
                    return true;
                } else if (is_hover_vseparator_up_) {
                    is_dragging_vseparator_up_ = true;
                    return true;
                } else if (is_hover_vseparator_down_) {
                    is_dragging_vseparator_down_ = true;
                    return true;
                }
            }
            if (!isDragging()) {
                return false;
            }
            if (is_dragging_hseparator_) {
            // y direction movement
                bias_y_ += event.mouse().y - hseparator_box_.y_min;
            } else {
            // x direction movement
                bias_x_ += event.mouse().x - vseparator_up_box_.x_min;
            }
            return true;
        } 


        bool Resizable4Blockbase::isDragging() {
            return (is_dragging_hseparator_ || is_dragging_vseparator_up_ || is_dragging_vseparator_down_ );
        };
        
        Element Resizable4Blockbase::getVSeparator() {
            return (is_hover_vseparator_up_ || is_hover_vseparator_down_) ? options_.separator_hover_func() : options_.separator_func();
        };

        Element Resizable4Blockbase::getHSeparator() {
            return (is_hover_hseparator_) ? options_.separator_hover_func() : options_.separator_func();
        };

        Component Resizable4Block(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options) {
            return Make<Resizable4Blockbase>(std::move(block1), std::move(block2), std::move(block3), std::move(block4), screen, std::move(options));
        }
    }
}