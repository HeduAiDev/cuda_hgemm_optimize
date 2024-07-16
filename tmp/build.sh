cd ..
xmake build --root tui_tool_sets
xmake install -o tmp --root tui_tool_sets
cd tmp
g++ -I./include test_main.cpp -o output -L./lib -ltui_tool_sets
