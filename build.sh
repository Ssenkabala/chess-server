#!/bin/bash
echo "Compiling engine..."
g++ -O3 -o engines/engine engine_src/engine.cpp
chmod +x engines/engine
echo "Done."
