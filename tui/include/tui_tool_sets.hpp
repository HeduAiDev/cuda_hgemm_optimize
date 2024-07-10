#pragma once

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

namespace tui {
    namespace component {
        using namespace ftxui;

        struct Resizable4BlockOptions {
            Element placeholder_block1 = nullptr;
            Element placeholder_block2 = nullptr;
            Element placeholder_block3 = nullptr;
            Element placeholder_block4 = nullptr;
            std::function<Element()> separator_func = [] { return separator(); };
            std::function<Element()> separator_hover_func = [] { return separatorHeavy(); };
            Resizable4BlockOptions() = default;
            // Custom copy constructor
            Resizable4BlockOptions(const Resizable4BlockOptions& other)
                : placeholder_block1(other.placeholder_block1),
                placeholder_block2(other.placeholder_block2),
                placeholder_block3(other.placeholder_block3),
                placeholder_block4(other.placeholder_block4),
                separator_func(other.separator_func),
                separator_hover_func(other.separator_hover_func) {}
            };

        class Resizable4Blockbase : public ftxui::ComponentBase {
            public:
                enum class Direction {
                    Horizontal,
                    Vertical
                };
                // block1 | block2
                // block3 | block4
                explicit Resizable4Blockbase(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options = Resizable4BlockOptions());
                
                Element Render() override;

                bool OnEvent(Event event) override;

                bool OnMouseEvent(Event event);

                bool isDragging();

                Element getVSeparator();
                Element getHSeparator();


                
                
            private:
                Component block1_;
                Component block2_;
                Component block3_;
                Component block4_;
                ScreenInteractive& screen_;
                Box vseparator_up_box_;
                Box vseparator_down_box_;
                Box hseparator_box_;
                bool is_hover_hseparator_ = false;
                bool is_hover_vseparator_up_ = false;
                bool is_hover_vseparator_down_ = false;
                bool is_dragging_hseparator_ = false;
                bool is_dragging_vseparator_up_ = false;
                bool is_dragging_vseparator_down_ = false;
                int bias_x_ = 0;
                int bias_y_ = 0;
         
                Resizable4BlockOptions options_;
        };

        struct RadioFrameOptions : public RadioboxOption {
            std::string title_regx = "selected: %s";
            int max_width = 200;
            int max_height = 200;
            int min_width = 0;
            int min_height = 0;
            RadioFrameOptions() = default;
        };

        class RadioFrameBase : public ftxui::ComponentBase, public RadioFrameOptions {
            public:
                explicit RadioFrameBase(RadioFrameOptions& options = RadioFrameOptions());
                Element Render() override;
            private:
                Component content_;
        };




        Component Resizable4Block(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options);
        Component RadioFrame(RadioFrameOptions options);
        Component RadioFrame(ConstStringListRef entries, int* selected, RadioFrameOptions options = RadioFrameOptions());
    }
}