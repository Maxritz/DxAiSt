# AGENTS.md — DXAiT Development Guidelines

## Project Overview
DXAiT is a DirectX 12 (D3D12) GPU compute fabric and SDK for Windows.
- **Language**: Pure C++ (C++20/C++23)
- **API Scope**: Modern C++ (`namespace dxait`, RAII, `ComPtr`, stdlib)
- **Shaders**: HLSL / Shader Model 6.0+ (`src/shaders/`)

## Build & Test Commands
```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 16
ctest --test-dir build -C Release --output-on-failure
```

## Mandatory Agent Principles & Skills
- **Caveman Skill**: ALWAYS ACTIVE for all agent responses (ultra-terse output, no filler).
- **Ponytail Skill**: Leverage for code review, audit, writing, and resolving tasks:
  - YAGNI (skip speculative features)
  - Pure modern C++ (C++20/C++23) & native D3D12 first
  - Minimal diffs, no unnecessary abstractions
  - Zero tolerance for stubs, TODOs, fake functions
- **Harness-First**: Build and run D3D12 CTest suite before and after any change.
- **Verification**: Run `ctest` or specific unit test binary after every modification.
