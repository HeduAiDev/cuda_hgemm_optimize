
add_rules("mode.debug", "mode.release")
includes("3rd", "tools")
set_languages("c++17")
add_requires("cutlass", "eigen", "ftxui", {system = false})
add_links('cublas')
add_cugencodes("sm_75")
set_policy("build.across_targets_in_parallel", false)

target("compilePrepare")
    set_kind("phony")
    on_build(function()
        import("core.base.task")
        task.run("render-headers")
    end)

target("algorithm")
    set_default(false)
    set_kind("phony")
    set_targetdir("dist/alg", {interface = true})
    add_packages("cutlass", {interface = true})
    add_includedirs("$(buildir)/include", {interface = true})
    add_deps("compilePrepare")

target("tui_tool_sets")
    set_default(false)
    set_kind("static")
    add_files("tui/Component/*.cpp")
    set_targetdir("$(buildir)/lib")
    add_includedirs("tui/include")
    add_packages("ftxui")
    add_deps("algorithm")

target("TUI")
    set_kind("binary")
    add_files("tui/*.cu")
    add_includedirs("tui/include")
    set_targetdir("dist/tui")
    add_packages("ftxui")
    add_deps("algorithm", "tui_tool_sets")
    -- add_ldflags("-static", {force = true})



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