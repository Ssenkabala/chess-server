#!/usr/bin/env bash
# build_wasm.sh — Compile SenkabalaIII to WebAssembly using Emscripten
#
# Prerequisites:
#   1. Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html
#      git clone https://github.com/emscripten-core/emsdk.git
#      cd emsdk && ./emsdk install latest && ./emsdk activate latest
#      source ./emsdk_env.sh
#
#   2. Run this script from the directory containing engine_wasm.cpp:
#      chmod +x build_wasm.sh
#      ./build_wasm.sh
#
# Output:
#   senkabala.js   — Emscripten loader (include in HTML)
#   senkabala.wasm — WebAssembly binary (served alongside JS)
#
# Deploy both files to your Railway static directory (same folder as index.html).
# Then include in index.html:
#   <script src="/senkabala.js"></script>
#   <script src="/senkabala_wasm.js"></script>

set -e

echo "[build_wasm] Compiling SenkabalaIII → WebAssembly..."

emcc engine_wasm.cpp \
  -O3 \
  -std=c++17 \
  -fno-exceptions \
  -s WASM=1 \
  -s "EXPORTED_FUNCTIONS=['_engine_init','_engine_best_move','_engine_analyse']" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=134217728 \
  -s MAXIMUM_MEMORY=536870912 \
  -s STACK_SIZE=4194304 \
  -s ENVIRONMENT=web \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=SenkabalaModule \
  -s NO_EXIT_RUNTIME=1 \
  -o senkabala.js

echo "[build_wasm] Done."
echo "  Output: senkabala.js + senkabala.wasm"
echo "  Total size: $(du -sh senkabala.wasm | cut -f1) (WASM binary)"
echo ""
echo "Deploy both files to your Railway static directory."
echo "Then add to index.html:"
echo '  <script src="/senkabala.js"></script>'
echo '  <script src="/senkabala_wasm.js"></script>'
