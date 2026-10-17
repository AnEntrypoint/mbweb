# PRD-02: CPU-only, single-thread WASI core build

## Problem
The CMake project targets native hosts. It enables Vulkan, llamafile (native assembly),
MuJoCo physics, ggml CPU variant loading (dynamic backend modules), and sanitizers. None of
these exist in a WebAssembly sandbox. The Go demo and CLI are out of scope.

## Requirements
1. Add a CMake toolchain file `cmake/wasi-p3.cmake` that uses the pinned SDK from PRD-01.
2. Configure with: `MOTIONBRICKS_ENABLE_GGML=ON`, `MOTIONBRICKS_ENABLE_VULKAN=OFF`,
   `MOTIONBRICKS_ENABLE_LLAMAFILE=OFF`, `MOTIONBRICKS_CPU_ALL_VARIANTS=OFF`,
   `MOTIONBRICKS_ENABLE_PHYSICS=OFF`, `MOTIONBRICKS_DOWNLOAD_MODELS=OFF`,
   `MOTIONBRICKS_BUILD_TESTS=OFF`, `BUILD_SHARED_LIBS=OFF`.
3. GGML threads off (`GGML_OPENMP=OFF`, no pthreads) so that a single-thread component works.
4. Build only `motionbricks_static` plus the inference C ABI (`inference.h`, `motionbricks.h`).
   Exclude `sonic*`, `physics.h`, and `src/cli.cpp`.
5. Disable exceptions and RTTI only if ggml or the sources need it to link; otherwise keep the
   project's C++23 flags.
6. Link `-Wl,--export-dynamic`-free: export only the `mb_*` C ABI symbols.

## Acceptance
- `cmake --preset wasi-p3` configures and builds `libmotionbricks.a` for wasm32.
- A native test harness built from the same sources (run with `ctest` on the host) still
  passes the reference parity check for one G1 inference call.

## Risks
- Peak memory: F32 weights are ~733 MB; wasm32 linear memory is limited to 4 GB, so this fits
  but with little headroom for activations. Measure in PRD-05.
- Some `src/*.cpp` may assume POSIX file I/O; the GGUF loader may need a memory-buffer entry point.
