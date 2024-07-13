xmake build tui_tool_sets
xmake install -o ./ tui_tool_sets
cl.exe /I ./include test_main.cpp /Fe:output.exe /link /LIBPATH:./lib tui_tool_sets.lib