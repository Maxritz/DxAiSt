# CLAUDE.md — DXAiT Developer Reference & Principles

# Principles & Skills

- **Caveman (ALWAYS ACTIVE)** — Every response must use caveman skill (`~/.config/opencode/skills/caveman/SKILL.md`): ultra-terse output, drop filler/articles/hedging, keep code/errors/technical terms verbatim. No narration of tool calls.
- **Ponytail (ALWAYS ACTIVE)** — Leverage ponytail skill (`~/.cache/.../ponytail`) to review, audit, write, and resolve code:
  1. Does it need to exist at all? (YAGNI)
  2. Pure modern C++ (C++20/C++23) & native D3D12 first (`namespace dxait`, RAII, stdlib).
  3. Shortest working diff wins.
  4. Zero unrequested abstractions, zero boilerplate.
  5. Mark deliberate simplifications with `// ponytail:` comments.

## Why These Rules Exist

Without them, we waste hours debugging race conditions (trace catches), buffer overflows (truth table catches), wrong format conversions (decision tree catches), GPU hangs from load spikes (load check catches), and unnecessary token spends on source file reads.

# GPU Backend Development Rules (MANDATORY)

These rules apply specifically to DXAiT D3D12 / HLSL compute development:

- **Harness-first: ALWAYS build and run the targeted test suite before merging any code change into main.** Fix a bug? Build+run CTest suite covering the change. Add a feature? Build+run all tests. If the harness doesn't pass, the change does not go into main.
- **Test then main** — never push untested changes to production code paths.
- **Kernel/shader registry must match the build system** — `CMakeLists.txt` governs shader compilation (SM 6.0+ via `dxc.exe`). Any new HLSL shader source file (`.hlsl`) must be registered in `CMakeLists.txt` and added to the runtime PSO lookup table.
- **Command/queue lifecycle discipline** — D3D12 command list / allocator reset semantics: reset/reuse via documented fence primitive, do not reallocate or free memory in hot paths, skip allocator Reset on first use, full Reset after execution.

## Test Coverage Map (DXAiT)

| Test | Covers |
|------|--------|
| `test_device_init` | Device creation, adapter enumeration, feature level check |
| `test_cmd_lifecycle` | Command list/allocator reset, queue execution, fence sync |
| `test_tensor_alloc` | D3D12 heap allocation, upload/readback buffers, UAV state |
| `test_compute_dispatch` | PSO creation, root signature binding, compute dispatch correctness |

# Truth/False Logic Validation (MANDATORY)

**NO CODE WRITES WITHOUT A TRUTH TABLE FIRST.**

Before implementing any feature, fix, or change:

1. **Write the decision tree** — every `if`, every branch, every condition
2. **Write the truth table** — every condition evaluated against actual data
3. **Find the bugs** — trace through with real values, not assumptions
4. **Only then code** — implement after the trace proves correctness

## Format

```
## TRACE: <short name of the thing being traced>

### Input State
| Variable      | Value | Source                     |
|---------------|-------|----------------------------|
| <var>         | <val> | <where it comes from>      |

### Decision Tree
if (<condition>):
  TRUE → <path>
  FALSE → <path>

### Truth Table
| Condition                                           | Expected        | Actual          | PASS? |
|-----------------------------------------------------|-----------------|-----------------|-------|
| <condition 1>                                        | <expected>      | <actual>        | ✅ |
| <condition 2>                                        | <expected>      | <actual>        | ✅ |

### Race Conditions
- [ ] No read-before-write on shared data
- [ ] Barriers/synchronisation between producer and consumer
- [ ] No buffer overflow (check sizes)

### Load Conditions
- [ ] GPU load < 80%
- [ ] VRAM usage < 80%
- [ ] No dispatch storms (>1M workgroups/threadgroups)

### VERDICT: <PASS/FAIL — one line, and which backend/hardware this was traced against>
```

## Flow Chart Tracing for Issue Diagnosis (MANDATORY)

When debugging or analysing any problem, produce a directed flow chart BEFORE reading full source files.

### Format

```
## FLOW: [Issue Name]

[Entry Point]
    │
    ├── TRUE → [Branch A] → [Call X] → [Check Y]
    │                                  ├── PASS → [Output Z]
    │                                  └── FAIL → [Error Handler] ✗ BUG
    │
    └── FALSE → [Branch B] → [Default Path]
```

- Trace the happy path AND at least one error path
- Annotate each decision point with actual vs expected values
- Split flows longer than 15 nodes into sub-flows
- Mark every divergence between expected and actual with `✗`

# Stub / Placeholder / Mock / Dead-Code Audit Protocol (MANDATORY)

Whenever this codebase is modified, assume there are hidden stubs, placeholders, mock implementations, and un-wired modules. Do not report a feature as integrated until this protocol passes.

## 1. Audit Checklist

Before finishing any task, run these checks and report the result:

1. Search all new or changed files for: `TODO`, `FIXME`, `XXX`, `stub`, `placeholder`, `mock`, `fake`, `dummy`, `noop`, `NOT IMPLEMENTED`, `unimplemented`, `return 0; // TODO`, `return false; // TODO`.
2. For every new header/module added under `include/dxait/` or `src/`, verify it is included from a production source file and at least one symbol defined in it is invoked.
3. For every new HLSL shader added (`.hlsl`), verify:
   - it is listed in CMakeLists.txt,
   - the generated compiled DXIL output exists,
   - it is bound to a pipeline/dispatch path in the runtime, and
   - a real dispatch path uses that pipeline for real tensor shapes.
4. For every new helper/wrapper added, verify it is included and used in a production source file.

## 2. Severity Levels

| Level | Meaning | Example |
|-------|---------|---------|
| cosmetic | Comment-only; no executable effect | `// TODO: pick a better heuristic` |
| fallback | Real fallback exists; new path is optional | Disabled env-gated code that falls back to existing op |
| untested | Code path compiles but is never exercised by current tests | Kernel used only when a flag is set but no test sets it |
| dead | Included/compiled but never invoked | Header with helpers that nothing calls |
| mock | Returns fake results or empty implementation and may be misused | Function that claims to quantise but just copies data |

## 3. Required Report Format

For every stub/placeholder/mock/unwired-module found, report:
- File and line number(s).
- Severity level.
- What it currently does vs. what it should do.
- Implementation choice: **A. Remove**, **B. Wire and test**, or **C. Guard and warn**.

## 4. Zero-Tolerance Rule for Main Code

- Never write or keep code you do not fully understand, cannot explain, and have not validated end-to-end. No slacking, skipping, stubs, placeholders, fake/no-op functions, or "TODO" branches.
- Main source (`src/`, `include/`) must never contain stubs.
