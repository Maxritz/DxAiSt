# DXAiT — Vulkan 1.4 ↔ D3D12 API Mapping & Extension Specification

## 1. Core API & Handle Mapping

| Concept | Vulkan (SDK 1.4.357.0) | Direct3D 12 (Agility SDK 1.720) | DXAiT Modern C++ API |
| :--- | :--- | :--- | :--- |
| **Instance / Factory** | `VkInstance` | `IDXGIFactory6` | `dxait::Adapter::enumerate()` |
| **Physical Adapter** | `VkPhysicalDevice` | `IDXGIAdapter1` | `dxait::Adapter` |
| **Logical Device** | `VkDevice` | `ID3D12Device` | `dxait::Device` |
| **Execution Queue** | `VkQueue` | `ID3D12CommandQueue` | `dxait::Queue` |
| **Command Pool / Allocator** | `VkCommandPool` | `ID3D12CommandAllocator` | Internal RAII Pool |
| **Command Buffer / List** | `VkCommandBuffer` | `ID3D12GraphicsCommandList` | Command List Wrapper |
| **Fence / Timeline Semaphore**| `VkSemaphore` (Timeline) / `VkFence` | `ID3D12Fence` | `dxait::Fence` |
| **Memory Allocation** | `VkDeviceMemory` (`vkAllocateMemory`)| `ID3D12Heap` / Committed Resource | `dxait::Buffer` |
| **Buffer Resource** | `VkBuffer` | `ID3D12Resource` | `dxait::Buffer` |
| **Shader Bytecode** | SPIR-V (`VkShaderModule`) | DXIL (`ID3D12RootSignature` + PSO) | DXC SM 6.6+ DXIL |
| **Pipeline State** | `VkPipeline` (Compute) | `ID3D12PipelineState` | Pipeline Cache |
| **Descriptor Set / Heap** | `VkDescriptorSet` | `ID3D12DescriptorHeap` | Bindless Heap Manager |

---

## 2. Advanced & Underutilized Extension Mapping Matrix

High-value GPU hardware extensions available in Vulkan 1.4 & D3D12 (SM 6.6 - 6.8) that are underutilized by conventional AI/ML runtimes:

| Compute Feature | Vulkan Extension / Core | D3D12 Agility / SM Equivalent | AI/ML Impact & Advantage |
| :--- | :--- | :--- | :--- |
| **Matrix Hardware (WMMA/MMA)** | `VK_KHR_cooperative_matrix` | Shader Model 6.8 Wave Matrix | **+40-47% FP16 GEMM/Attention Decode** speedup on RDNA4 / Ada / Blackwell |
| **64-bit Device Pointers** | `VK_KHR_buffer_device_address` | SM 6.6 Resource Descriptor Heap (`ResourceDescriptorHeap[index]`) | Direct pointer arithmetic without descriptor table re-binding overhead |
| **Push Constants / Descriptors** | `VK_KHR_push_descriptor` | D3D12 Root Constants / Root Descriptors | Zero-allocation per-dispatch scalar arguments (shape, stride, quantization scales) |
| **Subgroup / Wave Uniformity** | `VK_KHR_shader_subgroup_uniform_control_flow` | SM 6.6 `WaveIsFirstLane()`, `WaveReadLaneFirst()` | Eliminates warp divergence in quantized GEMV decode kernels |
| **Timeline Synchronization** | `VK_KHR_timeline_semaphore` | D3D12 Monotonic Timeline `ID3D12Fence` | Lock-free multi-queue stream synchronization without CPU roundtrips |
| **Indirect Execution Graphs** | `VK_NV_device_generated_commands_compute` | SM 6.8 D3D12 Work Graphs / ExecuteIndirect | On-GPU dynamic loop dispatch (decoding, speculative decoding, dynamic graph execution) |
| **Extended Precision Math** | `VK_KHR_shader_float_controls` | SM 6.6 Native FP16 / BF16 / INT8 Packing (`dot4_add_i8`, `v_dot2_f32_f16`) | Double throughput on RDNA2 (`sdot4`/`sudot4`) and RDNA4 |
