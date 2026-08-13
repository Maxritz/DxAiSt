# DXAiT — Complete DirectX 12 Shader Model Specifications (SM 6.0 - SM 6.8)

## 1. Shader Model Evolution Matrix

| Shader Model | Introduced | Minimum Target Profile | Key Language & Runtime Features | DXC Compiler Flags | Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shader Model 6.0** | D3D12 Launch | `cs_6_0` | Base Wave Intrinsics (`WaveActiveSum`, `WaveActiveBallot`, `WaveReadLaneAt`, `WaveGetLaneIndex`) | `-T cs_6_0` | All D3D12 GPUs |
| **Shader Model 6.1** | Windows 10 1709 | `cs_6_1` | `SV_ViewID`, `SV_Barycentrics`, Multiview rendering | `-T cs_6_1` | D3D12 FL 12_1+ |
| **Shader Model 6.2** | Windows 10 1803 | `cs_6_2` | Native 16-bit Float (`float16_t`) and Int (`int16_t`) SIMD vectors | `-T cs_6_2 -enable-16bit-types` | AMD RDNA1/2/3/4, NV Pascal+ |
| **Shader Model 6.3** | Windows 10 1809 | `cs_6_3` | DirectX Raytracing 1.0 (DXR) state objects & Ray Generation Shaders | `-T cs_6_3` | D3D12 DXR Tier 1.0 |
| **Shader Model 6.4** | Windows 10 1903 | `cs_6_4` | Native 4-element 8-bit Integer Dot Products (`dot4_add_i8`, `dot4_add_u8`) | `-T cs_6_4` | AMD RDNA2/3/4, NV Turing+ |
| **Shader Model 6.5** | Windows 10 2004 | `cs_6_5` | DXR 1.1 Inline Ray Queries (`RayQuery`), Variable Rate Shading Tier 2 (VRS) | `-T cs_6_5` | AMD RDNA2/3/4, NV Ampere+ |
| **Shader Model 6.6** | Agility 1.4 | `cs_6_6` | Resource Descriptor Heap Indexing (`ResourceDescriptorHeap[]`), Wave Multi-Prefix (`WaveMultiPrefixSum`), Float Atomics (`InterlockedAdd`), Vector Packing | `-T cs_6_6` | AMD RDNA2/3/4, NV Ampere+ |
| **Shader Model 6.7** | Agility 1.606 | `cs_6_7` | Enhanced Barriers (`D3D12_BARRIER_GROUP`), Advanced Texture Ops, Extended Formats | `-T cs_6_7` | AMD RDNA3/4, NV Ada+ |
| **Shader Model 6.8** | Agility 1.720 | `cs_6_8` | **Wave Matrix Multiply Accumulate (WMMA)** (`WaveMatrix<T, M, N>`), **Work Graphs 1.0** (`ID3D12StateObject`) | `-T cs_6_8` | AMD RDNA4 (gfx1201), NV Blackwell |
| **Shader Model 6.9 (Experimental)** | Agility Preview | `cs_6_9` | **Work Graphs 1.1** (Dynamic Node Expansion, Multi-Entry Nodes), Native FP8 (`fp8_e4m3`, `fp8_e5m2`) Tensor Math | `-T cs_6_9` | AMD RDNA4+, Next-Gen GPUs |
| **Shader Model 6.10 (Future Specs)** | Experimental Roadmap | `cs_6_10` | Cooperative Wave Registers, Asynchronous Tensor Memory Copy Engines, Sub-Byte Quantized Atomic Operations | `-T cs_6_10` | Next-Gen AI/ML Fabric GPUs |

---

## 2. Dynamic Target Profile Selection in DXC JIT (`dxjit`)

```cpp
// DXC 1.10 Dynamic Shader Model Target Profile Selection
std::wstring select_shader_model_profile(const dxait::AdapterCaps& caps) {
    if (caps.shader_model >= D3D_SHADER_MODEL_6_8) {
        return L"cs_6_8"; // WMMA & Work Graphs available
    } else if (caps.shader_model >= D3D_SHADER_MODEL_6_6) {
        return L"cs_6_6"; // Descriptor Heap & Wave Multi-Prefix available
    } else if (caps.shader_model >= D3D_SHADER_MODEL_6_4) {
        return L"cs_6_4"; // dot4_add_i8 available
    }
    return L"cs_6_0"; // Legacy fallback
}
```

---

## 3. Shader Model 6.8 Wave Matrix Primitive API

```hlsl
// Shader Model 6.8 Wave Matrix API Example
#if __SHADER_TARGET_MAJOR >= 6 && __SHADER_TARGET_MINOR >= 8
WaveMatrix<float16_t, 16, 16> matA;
WaveMatrix<float16_t, 16, 16> matB;
WaveMatrix<float, 16, 16> matC;

WaveMatrixFill(matC, 0.0f);
WaveMatrixLoad(matA, bufferA, offsetA, strideA);
WaveMatrixLoad(matB, bufferB, offsetB, strideB);
WaveMatrixMultiplyAccumulate(matC, matA, matB, matC);
WaveMatrixStore(matC, bufferC, offsetC, strideC);
#endif
```
