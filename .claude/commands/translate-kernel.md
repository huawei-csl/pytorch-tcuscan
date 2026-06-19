---
description: "Translate an AscendC kernel header to PTO-ISA. Usage: /translate-kernel <path/to/kernel_foo.h>"
argument-hint: "<path/to/kernel_foo.h>"
allowed-tools: ["Agent", "Read", "Bash"]
---

You are implementing the `/translate-kernel` skill for the pytorch-tcuscan project.

## What this skill does

Translates an AscendC kernel header (`kernel_foo.h`) to its PTO-ISA equivalent (`kernel_pto_foo.h`), placed alongside the source file — matching the existing pattern in this repo (e.g. `src/kernels/kernel_pto_cube_reduce.h` is the PTO port of `src/kernels/kernel_cube_reduce.h`).

## Input

The argument is `$ARGUMENTS` — a file path to an AscendC kernel header that **must exist inside the current repository where Claude is running**. The path may be relative to the repo root or absolute, but it must resolve to a file within the repo. Examples:

```
src/kernels/kernel_histogram.h
/home/zouzias/github-repos/pytorch-tcuscan/src/kernels/kernel_scan_single_core.h
```

If the resolved path falls outside the current repository, tell the user and stop.

## Steps

1. **Resolve paths.** Given `$ARGUMENTS`:
   - If it doesn't end in `.h`, tell the user and stop.
   - Compute the absolute source path. If the path is relative, resolve it against the repo root (`/home/zouzias/github-repos/pytorch-tcuscan`).
   - Verify the resolved path is inside the current repository. If it falls outside the repo root, tell the user and stop.
   - Derive the output filename: replace the leading `kernel_` prefix in the basename with `kernel_pto_` (e.g. `kernel_histogram.h` → `kernel_pto_histogram.h`). Place the output in the same directory as the source.
   - If the output file already exists, warn the user and ask for confirmation before overwriting.

2. **Read the reference translation** to ground the agent's style:
   - Read `src/kernels/kernel_pto_cube_reduce.h` as a concrete reference for how PTO-ISA kernels look in this repo (PTO tile types, `TLOAD`/`TSTORE`/`TMATMUL`/`TSYNC` patterns, `pto::GlobalTensor` descriptors, namespace structure, etc.).

3. **Delegate to the translation agent.** Spawn the `translate-ascendc-to-pto-isa` subagent with a prompt that includes:
   - The absolute path of the source file (so the agent reads it).
   - The absolute path of the output file (so the agent writes it).
   - The path of the reference file (`src/kernels/kernel_pto_cube_reduce.h`) for style reference.
   - Explicit instructions for using the `npu-coding` MCP tools at each stage (see the **MCP tool usage** section below).
   - Instruction to write the final PTO-ISA kernel to the output file when done.

   Example agent prompt template:

   ```
   Translate the AscendC kernel at <SOURCE_PATH> into a PTO-ISA kernel and
   write the result to <OUTPUT_PATH>.

   Style reference: read src/kernels/kernel_pto_cube_reduce.h first to understand
   the PTO-ISA coding conventions used in this repo (tile types, TSYNC usage,
   GlobalTensor descriptors, pto::scalar helpers, namespace structure).

   ## MCP tool usage — MANDATORY

   You have access to the `npu-coding` MCP server. Use it at every step of the
   translation. Never guess instruction names or syntax — always look them up first.

   ### Step A — Survey available instructions
   Before touching any AscendC API, call:
   - `mcp__npu-coding__list_categories` — get the top-level PTO-ISA categories.
   - `mcp__npu-coding__list_instructions` — enumerate all available instructions
     so you know what exists.

   ### Step B — Map each AscendC API call
   For every AscendC call in the source (e.g. `DataCopy`, `Add`, `Duplicate`,
   `Gather`, `CompareScalar`, …):
   1. Call `mcp__npu-coding__search_instructions("<keyword>")` with the AscendC
      op name or a synonym (e.g. `"gather"`, `"add"`, `"duplicate"`).
   2. For each candidate returned, call `mcp__npu-coding__get_instruction("<name>")`
      to read its full spec (signature, semantics, constraints).
   3. Call `mcp__npu-coding__get_constraints("<name>")` to confirm the dtype and
      alignment requirements match the kernel's template parameters.
   4. Call `mcp__npu-coding__get_examples("<name>")` to verify the correct call
      pattern (argument order, tile descriptors, UB address layout).

   ### Step C — Low-level / intrinsic ops
   For pipe synchronisation (`set_flag`, `wait_flag`, `pipe_barrier`) and scalar
   helpers (`CeilDiv`, `min`, …):
   - Call `mcp__npu-coding__get_auxiliary_ops` for `set_flag` / `wait_flag` /
     `pipe_barrier` / event flag patterns.
   - Call `mcp__npu-coding__get_scalar_ops` for scalar arithmetic helpers.
   - Call `mcp__npu-coding__get_control_flow_ops` for loop and branching idioms.
   - Call `mcp__npu-coding__get_cpp_intrinsic("<name>")` for any C++ intrinsic
     that appears in the source but has no obvious PTO-ISA equivalent.

   ### Step D — Validate grammar and assembly
   After drafting each major section of the output:
   - Call `mcp__npu-coding__get_grammar` to verify the overall PTO-ISA statement
     structure is correct.
   - Call `mcp__npu-coding__get_assembly_format("<name>")` for any instruction
     whose binary encoding you are uncertain about.

   ### Step E — AscendC / CCE reference (for unfamiliar source APIs)
   If the source uses an AscendC API you don't recognise:
   - Call `mcp__npu-coding__ascendc_search_docs("<api_name>")` or
     `mcp__npu-coding__ascendc_search_api("<api_name>")` to read the authoritative
     AscendC definition before attempting a translation.
   - Call `mcp__npu-coding__cce_search_docs` / `mcp__npu-coding__cce_search_api`
     for lower-level CCE intrinsics.
   - Call `mcp__npu-coding__get_cce_intrinsics("<name>")` for CCE-level details.

   ## Reference PTO-ISA patterns from pto-kernels

   Use the two canonical examples below as style anchors for common patterns.

   ### Example 1 — simple vector unary (abs / elementwise op)
   Source: https://github.com/huawei-csl/pto-kernels/blob/main/csrc/kernel/kernel_abs.cpp

   Key patterns to reproduce:
   - `pto::Shape<1,1,1,1,DYNAMIC>` + `pto::Stride<1,1,1,1,1>` for `GlobalTensor`.
   - `Tile<TileType::Vec, T, 1, TILE_SIZE, BLayout::RowMajor, 1, DYNAMIC>` for UB tiles.
   - `TASSIGN(tile, UB_ADDR)` to bind a tile to a UB address slot.
   - Single-event double-buffer pipeline:
     - Initialise with `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0)` and
       `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0)` before the loop.
     - Inside loop: `wait_flag(PIPE_V, PIPE_MTE2)` → `TLOAD` → `set_flag(PIPE_MTE2, PIPE_V)` →
       `wait_flag(PIPE_MTE2, PIPE_V)` + `wait_flag(PIPE_MTE3, PIPE_V)` → compute →
       `set_flag(PIPE_V, PIPE_MTE2)` + `set_flag(PIPE_V, PIPE_MTE3)` →
       `wait_flag(PIPE_V, PIPE_MTE3)` → `TSTORE` → `set_flag(PIPE_MTE3, PIPE_V)`.
     - Drain remaining flags after the loop exits.
   - `set_mask_norm()` + `set_vector_mask(-1, -1)` at kernel entry.
   - `get_block_num()` / `get_block_idx()` for multi-core work distribution.
   - Entry-point guards: `#if defined(__DAV_VEC__)`.

   ### Example 2 — gather with triple-buffer and multi-event sync (csr_gather)
   Source: https://github.com/huawei-csl/pto-kernels/blob/main/csrc/kernel/kernel_csr_gather.cpp

   Key patterns to reproduce:
   - Multiple UB regions with compile-time address arithmetic:
     `constexpr uint32_t V_T_ADDR = 0`, `W_T_ADDR = V_T_ADDR + N_BUF * TILE_SIZE_IN_BYTES`, …
   - `static_assert(X_T_ADDR + TILE_SIZE_X_IN_BYTES <= UB_USABLE_BYTES, …)` to
     guard UB overflow at compile time.
   - Pre-loop bulk load of a large tile: `TLOAD(xTiles, xGlobal)` followed by
     `set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7)` to signal its completion.
   - Triple-buffer rotation via `stage = (stage + 1) % N_BUF` and per-stage
     address computation.
   - Rotating event pairs: `ev0 = (event_t)((stage % N_BUF) * 2)`,
     `ev1 = (event_t)((stage % N_BUF) * 2 + 1)`.
   - `TGATHER(wTiles, xTiles, idxTiles, tmpTiles)` — gather requires a tmp tile
     of the same size as the index tile (PTO-ISA 8.5.0 constraint; the dst
     address is reused as tmp).
   - `pipe_barrier(PIPE_V)` to enforce ordering between gather and mul inside
     the same PIPE_V.
   - `TMUL(zTiles, valTiles, wTiles)` for elementwise multiply after gather.
   - Exhaustive flag drain after the loop (one `wait_flag` per initialised event).

   ## Output requirements

   - Complete, compilable PTO-ISA C++ (`#include "kernel_utils.h"`, `using namespace pto;`).
   - **No C++ classes.** Express the kernel as a plain `extern "C"` function (or a
     free function under the `#if defined(__DAV_VEC__)` guard), not as a class or
     struct with member functions. Tile descriptors and UB addresses are local
     variables or `constexpr` constants, not class members.
   - Same algorithmic logic as the source (tiling strategy, loop bounds, work split).
   - No pseudocode, no TODO stubs.
   - Annotate any non-obvious AscendC → PTO-ISA mapping with a one-line comment.
   - If an AscendC API has no direct PTO-ISA equivalent, decompose it into
     available primitives and document the decomposition.
   - The output filename must contain `_pto_` (e.g. `kernel_cube_reduce.h` →
     `kernel_pto_cube_reduce.h`). Reject any output path that does not match this
     pattern.
   - Write the completed kernel to <OUTPUT_PATH>.
   ```

4. **Report outcome.** After the agent finishes, tell the user:
   - The output file path that was written.
   - Any AscendC APIs for which no direct PTO-ISA mapping was found (the agent should surface these).
   - A reminder to inspect the output and run the existing test suite.
