package("cutlass")
    set_kind("library", {headeronly = true})
    set_homepage("https://github.com/NVIDIA/cutlass")
    set_description("CUDA Templates for Linear Algebra Subroutines")

    add_urls("https://github.com/NVIDIA/cutlass.git")

    add_versions("v3.5.0", "ef6af8526e3ad04f9827f35ee57eec555d09447f70a0ad0cf684a2e426ccbcb6")
    add_versions("v3.4.1", "aebd4f9088bdf2fd640d65835de30788a6c7d3615532fcbdbc626ec3754becd4")
    add_versions("v3.2.0", "9637961560a9d63a6bb3f407faf457c7dbc4246d3afb54ac7dc1e014dd7f172f")
    
    set_installdir(path.join(os.projectdir(), '3rd', 'cutlass'))
    on_install(function (package)
        local source = os.dirs("cutlass*")
        if source and #source ~= 0 then
            os.cd(source[1])
        end
        os.cp("include", package:installdir())
    end)

    on_test(function (package)
        assert(package:has_cxxincludes("cutlass/cutlass.h", {configs = {languages = "c++17"}}))
    end)

package("eigen")

    set_kind("library", {headeronly = true})
    set_homepage("https://eigen.tuxfamily.org/")
    set_description("C++ template library for linear algebra")
    set_license("MPL-2.0")

    add_urls("https://gitlab.com/libeigen/eigen/-/archive/$(version)/eigen-$(version).tar.bz2",
             "https://gitlab.com/libeigen/eigen.git")
    add_versions("3.3.7", "685adf14bd8e9c015b78097c1dc22f2f01343756f196acdc76a678e1ae352e11")
    add_versions("3.3.8", "0215c6593c4ee9f1f7f28238c4e8995584ebf3b556e9dbf933d84feb98d5b9ef")
    add_versions("3.3.9", "0fa5cafe78f66d2b501b43016858070d52ba47bd9b1016b0165a7b8e04675677")
    add_versions("3.4.0", "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626")
    
    if is_plat("mingw") and is_subhost("msys") then
        add_extsources("pacman::eigen3")
    elseif is_plat("linux") then
        add_extsources("pacman::eigen", "apt::libeigen3-dev")
    elseif is_plat("macosx") then
        add_extsources("brew::eigen")
    end

    add_deps("cmake")
    add_includedirs("include")
    add_includedirs("include/eigen3")
    set_installdir(path.join(os.projectdir(), '3rd', 'Eigen', os.host()))

    on_install("macosx", "linux", "windows", "mingw", "cross", function (package)
        import("package.tools.cmake").install(package, {"-DBUILD_TESTING=OFF"})
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <iostream>
            #include <Eigen/Dense>
            using Eigen::MatrixXd;
            void test()
            {
                MatrixXd m(2,2);
                m(0,0) = 3;
                m(1,0) = 2.5;
                m(0,1) = -1;
                m(1,1) = m(1,0) + m(0,1);
                std::cout << m << std::endl;
            }
        ]]}, {configs = {languages = "c++11"}}))
    end)


package("ftxui")
    set_homepage("https://github.com/ArthurSonzogni/FTXUI")
    set_description(":computer: C++ Functional Terminal User Interface. :heart:")
    set_license("MIT")

    add_urls("https://github.com/ArthurSonzogni/FTXUI/archive/refs/tags/$(version).tar.gz",
             "https://github.com/ArthurSonzogni/FTXUI.git")
    add_versions("v5.0.0", "a2991cb222c944aee14397965d9f6b050245da849d8c5da7c72d112de2786b5b")
    add_versions("v4.1.1", "9009d093e48b3189487d67fc3e375a57c7b354c0e43fc554ad31bec74a4bc2dd")
    add_versions("v3.0.0", "a8f2539ab95caafb21b0c534e8dfb0aeea4e658688797bb9e5539729d9258cc1")

    add_deps("cmake")

    if is_plat("windows") then
        add_configs("shared", {description = "Build shared library.", default = false, type = "boolean", readonly = true})
    elseif is_plat("linux", "bsd") then
        add_syslinks("pthread")
    end

    add_links("ftxui-component", "ftxui-dom", "ftxui-screen")
    set_installdir(path.join(os.projectdir(), '3rd', 'ftxui', os.host()))

    on_install("linux", "windows", "macosx", "bsd", "mingw", "cross", function (package)
        local configs = {"-DFTXUI_BUILD_DOCS=OFF", "-DFTXUI_BUILD_EXAMPLES=OFF"}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        import("package.tools.cmake").install(package, configs)
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <memory>
            #include <string>

            #include "ftxui/component/captured_mouse.hpp"
            #include "ftxui/component/component.hpp"
            #include "ftxui/component/component_base.hpp"
            #include "ftxui/component/screen_interactive.hpp"
            #include "ftxui/dom/elements.hpp"

            using namespace ftxui;

            void test() {
                int value = 50;
                auto buttons = Container::Horizontal({
                  Button("Decrease", [&] { value--; }),
                  Button("Increase", [&] { value++; }),
                });
                auto component = Renderer(buttons, [&] {
                return vbox({
                           text("value = " + std::to_string(value)),
                           separator(),
                           gauge(value * 0.01f),
                           separator(),
                           buttons->Render(),
                       }) |
                       border;
                });
                auto screen = ScreenInteractive::FitComponent();
                screen.Loop(component);
            }
        ]]}, {configs = {languages = "c++17"}}))
    end)