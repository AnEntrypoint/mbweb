# Toolchain for the motion-bricks threaded core (wasm32-wasip1-threads). Unlike the p3 component path, this
# target has real pthreads: WebAssembly threads over one shared memory, spawned through wasi:thread-spawn.
# The page loads this core directly with JS workers and SharedArrayBuffer (cross-origin isolated pages).
#
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=<mbweb>/cmake/wasi-threads.cmake -DWASI_SDK_PREFIX=<sdk> ...

if(NOT WASI_SDK_PREFIX)
    if(DEFINED ENV{WASI_SDK_PREFIX})
        set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PREFIX}")
    else()
        message(FATAL_ERROR "Set WASI_SDK_PREFIX to the unpacked wasi-sdk 34 directory")
    endif()
endif()

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)
set(CMAKE_EXECUTABLE_SUFFIX .wasm)
set(triple wasm32-wasip1-threads)

set(CMAKE_C_COMPILER ${WASI_SDK_PREFIX}/bin/clang)
set(CMAKE_CXX_COMPILER ${WASI_SDK_PREFIX}/bin/clang++)
set(CMAKE_ASM_COMPILER ${WASI_SDK_PREFIX}/bin/clang)
set(CMAKE_AR ${WASI_SDK_PREFIX}/bin/llvm-ar)
set(CMAKE_RANLIB ${WASI_SDK_PREFIX}/bin/llvm-ranlib)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_ASM_COMPILER_TARGET ${triple})

# Atomics and bulk memory are what a shared memory needs; pthread enables the wasi-libc thread layer.
set(MB_THREAD_FLAGS "-pthread -matomics -mbulk-memory")
set(MB_WASI_EMULATION_DEFINES "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${MB_THREAD_FLAGS} -msimd128 ${MB_WASI_EMULATION_DEFINES}")
set(CMAKE_CXX_FLAGS_INIT "${MB_THREAD_FLAGS} -msimd128 ${MB_WASI_EMULATION_DEFINES} -fno-exceptions -include errno.h")
# The module imports one shared memory, so every thread sees the same weights and buffers.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread -Wl,--import-memory -Wl,--shared-memory -Wl,--export-memory -Wl,--max-memory=4294967296 -lwasi-emulated-signal -lwasi-emulated-process-clocks")

set(CMAKE_FIND_ROOT_PATH ${WASI_SDK_PREFIX}/share/wasi-sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
