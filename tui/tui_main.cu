

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


using namespace ftxui;
int main()
{
    auto screen = ScreenInteractive::Fullscreen();
    // kernel selector
    std::vector<std::string> kernel_available = {"kernel1", "kernel2", "kernel3"};
    int kernel_selected = 0;
    Component kernel_selector = Radiobox(&kernel_available, &kernel_selected);
    // config panel
    std::string config_m;
    std::string config_n;
    std::string config_k;
    std::string config_tile_m;
    std::string config_tile_n;
    std::string config_tile_k;
    std::string config_launch_cnt;

    InputOption input_style = InputOption::Spacious();
    Component input_config_m = Input(&config_m, "m");
    Component input_config_n = Input(&config_n, "n");
    Component input_config_k = Input(&config_k, "k");
    Component input_config_tile_m = Input(&config_tile_m, "tile m");
    Component input_config_tile_n = Input(&config_tile_n, "tile n");
    Component input_config_tile_k = Input(&config_tile_k, "tile k");
    Component input_config_launch_cnt = Input(&config_launch_cnt, "launch count");

    Component config_panel = Container::Vertical({
        Container::Horizontal({
            input_config_m,
            input_config_n,
            input_config_k,
        }),
        Container::Horizontal({
            input_config_tile_m,
            input_config_tile_n,
            input_config_tile_k,
        }),
        input_config_launch_cnt
    });

    Component config_panel_renderer = Renderer(config_panel, [&] {
        return vbox({
            hbox({ input_config_n -> Render(), filler(), input_config_m -> Render(), filler(), input_config_k -> Render() }),
            hbox({ input_config_tile_n -> Render(), filler(), input_config_tile_m -> Render(), filler(), input_config_tile_k -> Render() }),
            input_config_launch_cnt -> Render()
        }) | border;
    });

    Component block1 = Container::Vertical({
        kernel_selector,
        config_panel_renderer
    });

    Component block1_renderer = Renderer(block1, [&] {
        return vbox({
            kernel_selector -> Render(),
            config_panel_renderer -> Render()
        }) | flex;
    });

    Component block2 = Renderer([] { return text("block2") | center | flex;});
    Component block3 = Renderer([] { return text("block3") | center | flex;});
    Component block4 = Renderer([] { return text("block4") | center | flex;});

    Component menu1 = Container::Vertical({
        Container::Horizontal({
            block1_renderer,
            block2,
        }),
        Container::Horizontal({
            block3,
            block4
        })
    });


    Component menu1_renderer = Renderer(menu1, [&] {
        return vbox({
            hbox({
                block1_renderer -> Render() | border | size(WIDTH, EQUAL, screen.dimx() / 2),
                block2 -> Render() | border | flex
            }) | size(HEIGHT, EQUAL, screen.dimy() / 2),
            hbox({
                block3 -> Render() | border | size(WIDTH, EQUAL, screen.dimx() / 2),
                block4 -> Render() | border | flex
            }) | size(HEIGHT, EQUAL, screen.dimy() / 2)
        });
    });

    int tab_index = 0;
    std::vector<std::string> tab_entries = {
        "menu1", "menu2", "menu3"
    };
    auto tab_section = Menu(&tab_entries, &tab_index, MenuOption::HorizontalAnimated());
    auto tab_content = Container::Tab({
        menu1_renderer,
        block2,
        block3
    }, &tab_index);
    Component main_container = Container::Vertical({
        tab_section,
        tab_content
    });
    Component main_renderer = Renderer(main_container, [&] {
        return vbox({
            text("Demo") | bold | hcenter,
            tab_section -> Render(),
            tab_content -> Render()
        });
    });
    screen.Loop(main_renderer);
    return 0;
}