# Toolchain for the motion-bricks WASIp3 core (PRD-01, PRD-02).
#
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=<mbweb>/cmake/wasi-p3.cmake -DWASI_SDK_PREFIX=<sdk> ...
# The SDK is the pinned wasi-sdk 34 unpacked by scripts/fetch-wasi-sdk.sh.

if(NOT WASI_SDK_PREFIX)
    if(DEFINED ENV{WASI_SDK_PREFIX})
        set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PREFIX}")
    else()
        message(FATAL_ERROR "Set WASI_SDK_PREFIX to the unpacked wasi-sdk 34 directory")
    endif()
endif()

include("${WASI_SDK_PREFIX}/share/cmake/wasi-sdk-p3.cmake")

# ggml calls signal() and clock(); WASI only provides them through emulation libraries.
set(MB_WASI_EMULATION_DEFINES "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS")
# The core is exception-free: jco 1.39 cannot translate wasm exception handling, so the
# patches in patches/motion-bricks/ replace the try/throw sites this build compiles
# (src/error.hpp, src/neural_runtime.cpp, ggml/src/gguf.cpp) and SONIC is excluded.
# gguf.cpp uses errno without including it, which the WASI C++ headers do not provide.
set(MB_WASM_NOEH_FLAGS "-fno-exceptions")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${MB_WASI_EMULATION_DEFINES}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${MB_WASI_EMULATION_DEFINES} ${MB_WASM_NOEH_FLAGS} -include errno.h")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -lwasi-emulated-signal -lwasi-emulated-process-clocks")
