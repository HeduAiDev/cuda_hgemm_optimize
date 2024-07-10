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
        RadioFrameBase::RadioFrameBase(RadioFrameOptions& options) : RadioFrameOptions(options) {
            content_ = Radiobox(entries, &selected()) | vscroll_indicator | frame | size(WIDTH, GREATER_THAN, min_width) | size(WIDTH, LESS_THAN, max_width) | size(HEIGHT, GREATER_THAN, min_height) | size(HEIGHT, LESS_THAN, max_height); 
            Add(content_);
        };
        Element RadioFrameBase::Render() {
            return vbox({
                hbox({
                    text(std::string(title_regx).replace(title_regx.find("%s"), 2, entries[selected()])) | bold
                }),
                separator(),
                content_ -> Render()
            });
        }

        Component RadioFrame(RadioFrameOptions options) {
            return Make<RadioFrameBase>(options);
        }

        Component RadioFrame(ConstStringListRef entries, int * selected, RadioFrameOptions options) {
            options.entries = entries;
            options.selected = selected;
            return Make<RadioFrameBase>(options);
        } 
    }
}