
add_rules("mode.debug", "mode.release")
includes("3rd", "tools")
set_languages("c++17")
add_requires("cutlass", "eigen", "ftxui", {system = false})
add_links('cublas')
add_cugencodes("sm_75")
set_policy("build.across_targets_in_parallel", false)
add_moduledirs("tools")

target("compilePrepare")
    set_kind("phony")
    on_build(function()
        import("core.base.task")
        task.run("render-headers")
    end)

target("algorithm")
    set_default(false)
    set_kind("phony")
    add_defines("__CUDA__", {interface = true})
    set_targetdir("dist/alg", {interface = true})
    add_packages("cutlass", {interface = true})
    add_includedirs("$(buildir)/include", {interface = true})
    add_deps("compilePrepare", "tui_tool_sets")

target("tui_tool_sets")
    set_default(false)
    set_kind("static")
    -- set_policy("build.merge_archive", true)
    add_defines("__COMPILE__")
    add_files("tui/Component/*.cpp")
    add_files("tui/runable/*.cpp")
    set_targetdir("$(buildir)/lib")
    add_includedirs("tui/include", {public = true})
    add_headerfiles("tui/include/*|color_*")
    add_packages("ftxui")
    if (is_host("windows")) then
        add_cxxflags("/MT")
    end
    on_load(function(target) 
        import("detect.sdks.find_cuda")
        local cuda = find_cuda()
        if cuda then
            target:add("defines", "__CUDA__")
            target:add("includedirs", cuda.includedirs)
        end
        -- target:add("linkdirs", cuda.linkdirs)
    end)
    on_install(function(target)
        import("utils.archive.merge_staticlib")
        import("table")
        if not target:installdir() or not os.exists(target:installdir()) then
            raise("${color.error}no installdir find, use -o/--installdir to specify, eg: xmake install -o /path/to/install "..target:name())
        end
        local install_file = path.join(target:installdir(), "lib", path.filename(target:targetfile()))
        print("merge", table.join(target:pkgs().ftxui:libraryfiles(), target:targetfile()),"to", install_file)
        -- 合并静态库
        merge_staticlib(target, install_file, table.join(target:pkgs().ftxui:libraryfiles(), target:targetfile()))
        for _, file in ipairs(target:headerfiles()) do
            os.cp(file, path.join(target:installdir(), "include", path.filename(file)))
        end
    end)
    

target("TUI")
    set_kind("binary")
    add_files("tui/*.cu")
    add_defines("__COMPILE__")
    add_includedirs("tui/include")
    set_targetdir("dist/tui")
    add_packages("ftxui")
    add_deps("algorithm", "tui_tool_sets")
    if (is_host("windows")) then
        add_cxxflags("/MT")
    end
    if (is_host("linux")) then
        add_cxxflags("-static")
        add_ldflags("-static")
    end



for _, file in ipairs(os.files("test/*.cu")) do 
    local file_name_without_ext = path.filename(file):match("(.+)%..+$")
    target(file_name_without_ext)
        set_kind("binary")
        add_files(file)
        set_targetdir("dist/test")
        add_deps("algorithm")
end

for _, file in ipairs(os.files("tui/example/*.cu")) do 
    local file_name_without_ext = path.filename(file):match("(.+)%..+$")
    target(file_name_without_ext)
        set_kind("binary")
        add_files(file)
        add_includedirs("tui/include")
        set_targetdir("dist/example")
        add_packages("ftxui")
        add_deps("algorithm")
end