
-- add_rules("mode.debug", "mode.release")
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

target("TUI")
    set_kind("binary")
    add_files("tui/*.cu")
    add_includedirs("tui/include")
    set_targetdir("dist/tui")
    add_packages("ftxui")
    -- add_deps("algorithm")



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