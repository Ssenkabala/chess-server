#!/bin/bash
echo "Compiling SenkabalaIII for Linux..."
g++ -O3 -std=c++17 -o engines/engine engine_src/engine.cpp -lpthread
chmod +x engines/engine
echo "Done."