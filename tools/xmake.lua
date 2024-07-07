add_moduledirs("tools")

task("render-headers")
    set_menu {usage = "xmake render-headers", description = "Adding specific custom syntax support for Header Files", 

    -- k：单纯的 key 类型参数，例如：-k/--key 这种传参方式，通常用于表示 bool 值状态。
    -- kv：键值类型传参，例如：-k value 或者 --key=value 的传参方式，通常用于获取指定参数值。
    -- v：单个值传参，例如：value，通常定义在所有参数的最尾部。
    -- vs：多值列表传参，也就是刚刚我们配置的参数类型，例如：value1 value2 ... 的传参方式，可以获取多个参数值作为列表输入。
    options = {
        {'f', "force", "k", nil, "force rerender"}
    }}
    set_category("plugin")
    on_run(function ()
        import("core.base.option")
        import("ansi_colors")
        import("core.project.config")
        for _, file in ipairs(os.files("include/**")) do
            local dist_file = path.join(config.buildir() , file)
            -- 仅在源文件修改后重新render
            if (not option.get('force') and os.exists(dist_file) and os.mtime(file) < os.mtime(dist_file)) then
                return
            end
            print("Render Headerfile" ,file)
            os.cp(file, dist_file)
            -- ansi颜色代码支持
            ansi_colors(dist_file)
        end
    end)