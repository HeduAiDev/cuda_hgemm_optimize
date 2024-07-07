    function ansi_color_syntax_support(sourcefile)
        -- Read the source file
        local file = io.open(sourcefile, "r")
        if not file then
            return
        end
        local content = file:read("*a")
        file:close()

        -- support ${blue} syntax in string
        content = content:gsub("%${green}", "\27[32m")
        content = content:gsub("%${red}", "\27[31m")
        content = content:gsub("%${blue}", "\27[34m")
        content = content:gsub("%${default}", "\27[0m")

        -- Write the modified content back to the file
        file = io.open(sourcefile, "w")
        if not file then
            return
        end
        file:write(content)
        file:close()
    end


    function main(file) 
        ansi_color_syntax_support(file)
    end