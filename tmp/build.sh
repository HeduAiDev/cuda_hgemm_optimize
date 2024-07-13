xmake build tui_tool_sets
xmake install -o ./ tui_tool_sets
g++ -I./include test_main.cpp -o output -L./lib -ltui_tool_sets
