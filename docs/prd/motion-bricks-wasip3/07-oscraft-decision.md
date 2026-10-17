# PRD-07: Decision: no shared libraries with ../oscraft

## Decision
Do not share libraries with `/workspace/oscraft`.

## Why
- oscraft is a Rust deterministic RTS simulation (`sim`, `mpq`, `data`, `server`, `spike`
  crates) with a Lua 5.4 mod runtime.
- motion-bricks is C++23 with a GGML inference core and a C ABI.
- The only overlap is the wasm build pattern, which is a recipe, not a library. Coupling the
  two would tie unrelated release cycles together.

## Revisit when
- oscraft gains a wasm build that motion-bricks can reuse as a toolchain step, or a shared
  scripting need arises.
