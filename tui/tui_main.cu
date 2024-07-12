

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
using namespace tui::component;

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
    tui::component::RadioFrameOptions radioframe_options;
    radioframe_options.max_height = 5;
    radioframe_options.title_regx = "kernel:%s";
    Component kernel_selector = tui::component::RadioFrame(&kernel_available, &kernel_selected, radioframe_options) | border;

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

    auto input_transform = [](InputState state)
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


    auto input_style = [](Element ele) { return ele | size(WIDTH, LESS_THAN, 12) | size(WIDTH, GREATER_THAN, 7) | size(HEIGHT, LESS_THAN, 5); };
    auto label_style = [](Element ele) { return ele | align_right | vcenter ;};

    auto input_cell = [&] (std::string label, ftxui::StringRef constent, std::string placeholder, tui::component::InputType inputType, std::function<Element(InputState)> transform)
    { 
        tui::component::InputElementConfig input_config;
        input_config.label = label;
        input_config.input_type = inputType;
        input_config.placeholder = placeholder;
        input_config.content = std::move(constent);
        input_config.transform = transform;
        input_config.input_style = input_style;
        input_config.label_style = label_style;
        return input_config;
    };
    

    Component input_form =  tui::component::InputForm({
        {
            input_cell("M :", &config_m, "M", tui::component::InputType::Number, input_transform),
            input_cell("N :", &config_n, "N", tui::component::InputType::Number, input_transform),
            input_cell("K :", &config_k, "K", tui::component::InputType::Number, input_transform),
        },
        {
            input_cell("TileM :", &config_tile_m, "TileM", tui::component::InputType::Number, input_transform),
            input_cell("TileN :", &config_tile_n, "TileN", tui::component::InputType::Number, input_transform),
            input_cell("TileK :", &config_tile_k, "TileK", tui::component::InputType::Number, input_transform),
        },
    });
    Component input_config_launch_cnt = Input(&config_launch_cnt, "launch count"); 
    
    input_form -> ChildAt(0) -> Add(input_config_launch_cnt);
    Component config_panel = Renderer(input_form, [&] {
        return window(text("config") | hcenter | bold, vbox({
            input_form -> Render(),
            hbox({label_style(text("Launch cnt :")), input_style(input_config_launch_cnt -> Render())})
        }))  | xflex_grow;
    });

    Component block1 = Renderer(
        Container::Vertical({
            kernel_selector,
            info_block,
            config_panel
        }),
        [&] {
        return vbox({
            hbox({kernel_selector -> Render(), info_block -> Render() | flex}),
            config_panel -> Render(),
        }) | flex;
    });

    /////////////////////////////////////////////////////////////////////
    //block2
    /////////////////////////////////////////////////////////////////////

    int M = 64;
    int N = 64;
    int K = 64;

    float* matrix_b_ptr = new float[K * N];
    float focus_M = 0.5f;
    float focus_N = 0.5f;
    ::tui::component::MatrixFrameOptions<float> matrix_b_options;
    matrix_b_options.focus_x = &focus_N;
    
    matrix_b_options.element_style =  tui::component::MatrixFrameOptionsCommonElementStyle::empty_style() | tui::component::MatrixFrameOptionsCommonElementStyle::mark_point_trace(10, 20, Color::Blue1, Color::Red1)
    | MatrixFrameOptionsCommonElementStyle::mark_sub_matrix(1,1,5,7,Color::Red3);
    matrix_b_options.label_marks.push_back({"col", 20, Color::Red1});
    matrix_b_options.label_marks.push_back({MatrixFrameOptionsLabelMark::LabelType::Row, 10, Color::Red1});
    auto block2 =  ::tui::component::MatrixFrame(matrix_b_ptr, K, N, matrix_b_options);
    block2 = Renderer(block2, [=]{ return window(text("MatrixB") | hcenter | bold, block2->Render());});

    ::tui::component::MatrixFrameOptions<float> matrix_a_options;
    matrix_a_options.focus_y = &focus_M;
    float* matrix_a_ptr = new float[M * K];
    Component block3 = ::tui::component::MatrixFrame(matrix_a_ptr, M, K, matrix_a_options);
    block3 = Renderer(block3, [=]{ return window(text("MatrixA") | hcenter | bold, block3->Render());});

    ::tui::component::MatrixFrameOptions<float> matrix_c_options;
    matrix_c_options.focus_x = &focus_N;
    matrix_c_options.focus_y = &focus_M;
    float* matrix_c_ptr = new float[M * N];
    Component block4 = ::tui::component::MatrixFrame(matrix_c_ptr, M, N, matrix_c_options);
    block4 = Renderer(block4, [=]{ return window(text("MatrixC") | hcenter | bold, block4->Render());});



    tui::component::Resizable4BlockOptions options;
    options.placeholder_block2 = text("Redraw matrix is inefficient") | center | bold;
    options.placeholder_block3 = text("Redraw matrix is inefficient") | center | bold;
    options.placeholder_block4 = text("Redraw matrix is inefficient") | center | bold;

    Component menu1_renderer = tui::component::Resizable4Block(block1, block2, block3, block4, screen, options);

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