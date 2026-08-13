# DXAiT — Agility SDK vs System D3D12 Shader Compilation Architecture

## 1. Overview & Dual-Mode Execution

DXAiT supports two shader execution and compilation modes:

| Mode | D3D12 Runtime Source | Max Shader Model | Key Features Available |
| :--- | :--- | :--- | :--- |
| **Agility SDK (1.720)** *(Default)* | `. \D3D12\D3D12Core.dll` | **Shader Model 6.8** | SM 6.8 Wave Matrix (WMMA), Work Graphs, Enhanced Barriers, DirectStorage 1.4 |
| **System D3D12 Fallback** | `C:\Windows\System32\d3d12.dll` | Shader Model 6.0 - 6.5 | Basic Compute Shaders, Descriptor Tables, Legacy Barriers |

---

## 2. Agility SDK Export Configuration

In modern C++ executables and DLLs, Agility SDK runtime enablement is declared at binary initialization via global exported constants:

```cpp
extern "C" {
    // Agility SDK 1.720 configuration
    __declspec(dllexport) extern const UINT D3D12SDKVersion = 720;
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
```

When D3D12CreateDevice is invoked, the OS D3D12 loader reads these exports and loads `.\D3D12\D3D12Core.dll`.

---

## 3. Shader Compilation Pipeline (DXC 1.10.2605.2)

1. **JIT / Dynamic Shader Compilation**:
   - `IDxcCompiler3` interface from `E:\DXllama\dxc-1.10.2605.2\bin\x64\dxcompiler.dll`.
   - Compiles HLSL source to DXIL bytecode targeting `cs_6_6`, `cs_6_7`, or `cs_6_8`.
   - Applies architecture-specific flags:
     - **RDNA2 (gfx1031)**: `-D WAVE_SIZE=64 -D ENABLE_DOT4=1`
     - **RDNA4 (gfx1201)**: `-D WAVE_SIZE=32 -D ENABLE_WMMA=1`

2. **AOT / Embedded Shader Registry**:
   - Pre-compiled DXIL bytecode blobs compiled during build via `dxc.exe`.
   - Embedded into statically linked header registries (`dxait_shader_registry.hpp`).
