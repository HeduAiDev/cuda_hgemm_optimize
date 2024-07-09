

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
#include <sstream>
 
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

using namespace ftxui;

int main()
{
    auto screen = ScreenInteractive::Fullscreen();
    /////////////////////////////////////////////////////////////////////
    //block1
    /////////////////////////////////////////////////////////////////////
    // kernel selector
    std::vector<std::string> kernel_available{};
    for(int i = 0; i < 20; i++) {
        kernel_available.push_back("kernel" + std::to_string(i));
    };
    int kernel_selected = 0;
    Component kernel_selector1 = Radiobox(&kernel_available, &kernel_selected) | vscroll_indicator | frame | size(HEIGHT, LESS_THAN, 5);
    Component kernel_selector = Renderer(kernel_selector1, [&] {
        return vbox({
            hbox({text("kernel:"), text(kernel_available[kernel_selected]) | bold }),
            separator(),
            kernel_selector1->Render()
        }) | border;
    });

    // runtime info
    bool ready = false;
    std::string info = 
        "simple usage: "
        " First choose a kernel."
        " Second input configs."
        " Third click run button.";
    Component run_button = Button("run", [&] {
        if (!ready) {
            info = "Please input config first";
            return;
        }
    }, ButtonOption::Animated());

    Component info_block  = Renderer(run_button ,[&] {
        return hbox({
            paragraph(info),
            separator(),
            run_button -> Render()
        }) | borderRounded;
    });

    // config panel
    std::string config_m;
    std::string config_n;
    std::string config_k;
    std::string config_tile_m;
    std::string config_tile_n;
    std::string config_tile_k;
    std::string config_launch_cnt;

    // InputOption input_style = InputOption::Spacious();
    InputOption input_option;
    input_option.transform = [](InputState state)
    {
        state.element |= borderRounded;
        state.element |= color(Color::White);

        if (state.is_placeholder)
        {
            state.element |= dim;
        }

        if (state.focused)
        {
            state.element |= bgcolor(Color::Black);
        }

        if (state.hovered)
        {
            state.element |= bgcolor(Color::GrayDark);
        }

        return state.element;
    };
    Component input_config_m = Input(&config_m, "M", input_option);
    Component input_config_n = Input(&config_n, "N", input_option);
    Component input_config_k = Input(&config_k, "K", input_option);
    Component input_config_tile_m = Input(&config_tile_m, "TileM", input_option);
    Component input_config_tile_n = Input(&config_tile_n, "TileN", input_option);
    Component input_config_tile_k = Input(&config_tile_k, "TileK", input_option);
    Component input_config_launch_cnt = Input(&config_launch_cnt, "launch count"); 
    
    input_config_m |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_n |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_k |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_tile_m |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_tile_n |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_tile_k |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
    input_config_launch_cnt |= CatchEvent([&](Event event) {
        return event.is_character() && !std::isdigit(event.character()[0]);
    });
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

    auto input = [](Element ele) { return ele | size(WIDTH, LESS_THAN, 12) | size(WIDTH, GREATER_THAN, 7) | size(HEIGHT, LESS_THAN, 5); };
    auto h = [](Element ele) { return ele | align_right | vcenter ;};
    Component config_panel_renderer = Renderer(config_panel, [&] {
        return window(text("config") | hcenter | bold, vbox({
            gridbox({
                {h(text(" M :")), input(input_config_m -> Render()), h(text(" N :")), input(input_config_n -> Render()), h(text(" K :")), input(input_config_k -> Render())},
                {h(text(" TileM :")), input(input_config_tile_m -> Render()), h(text(" TileN :")), input(input_config_tile_n -> Render()), h(text(" TileK :")), input(input_config_tile_k -> Render())},
            }),
            hbox({h(text("Launch cnt :")) , input(input_config_launch_cnt -> Render())})
        }))  | xflex_grow;

    });


    Component block1 = Container::Vertical({
        kernel_selector,
        info_block,
        config_panel_renderer
    });

    Component block1_renderer = Renderer(block1, [&] {
        return vbox({
            hbox({kernel_selector -> Render(), info_block -> Render() | flex}),
            config_panel_renderer -> Render(),
        }) | flex;
    });

    /////////////////////////////////////////////////////////////////////
    //block2
    /////////////////////////////////////////////////////////////////////

    auto make_box = [](std::string val) {
        return text(val) | center | size(WIDTH, EQUAL, 3) | size(HEIGHT, EQUAL, 1) | border;
    };

    auto make_grid = [&](int* ptr, int rows, int cols) {
        std::vector<Elements> crows;
        for (int i = 0; i < rows; i++) {
            std::vector<Element> ccols;
            for (int j = 0; j < cols; j++) {
                ccols.push_back(make_box(std::to_string(ptr[i * cols + j])));
            }
            crows.push_back(ccols);
        }
        return gridbox(crows);
    };

    float focus_x = 0.5f;
    float focus_y = 0.5f;
    SliderOption<float> slider_x_option = {&focus_x, 0.0f, 1.0f, 0.01f};
    SliderOption<float> slider_y_option = {&focus_y, 0.0f, 1.0f, 0.01f, Direction::Down};
    auto slider_x = Slider(slider_x_option);
    auto slider_y = Slider(slider_y_option);

    int* b_ptr = new int[128 * 128];

    Component block2 = Container::Vertical({
        slider_x,
        slider_y
    });

    Element matrix_b = make_grid(b_ptr, 128, 128);

    Component block2_renderer = Renderer(block2,
                                         [&]
                                         {
                                             return window(
                                                 text("matrixB") | hcenter | bold,
                                                 vbox({slider_x->Render() | size(HEIGHT, EQUAL, 1),
                                                       hbox({matrix_b | focusPositionRelative(focus_x, focus_y) | frame | flex,
                                                             slider_y->Render()})}));
                                         });
    Component block3 = Renderer([] { return text("block3") | center | flex;});
    Component block4 = Renderer([] { return text("block4") | center | flex;});



    tui::component::Resizable4BlockOptions options;
    // options.placeholder_block1 = text("Redraw matrix is inefficient") | center | bold;
    options.placeholder_block2 = text("Redraw matrix is inefficient") | center | bold;

    Component menu1_renderer = Resizable4Block(block1_renderer, block2_renderer, block3, block4, screen, options);

    int tab_index = 0;
    std::vector<std::string> tab_entries = {
        "menu1", "menu2", "menu3"
    };
    auto tab_section = Menu(&tab_entries, &tab_index, MenuOption::HorizontalAnimated());
    auto tab_content = Container::Tab({
        menu1_renderer,
        // block2,
        // block3
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