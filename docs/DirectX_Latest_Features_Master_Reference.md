# DXAiT — DirectX 12 Agility SDK & Shader Model 6.8 / 6.9 Latest Features Reference

## 1. Agility SDK 1.720 & Latest Feature Matrix

| DirectX 12 Feature | Agility SDK Version | Shader Model | HLSL / C++ API | AI/ML Subsystem Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Work Graphs 1.1** | Agility 1.720+ | Shader Model 6.8 / 6.9 | `ID3D12StateObject` / `BroadcastingLaunch` | **Zero-CPU overhead dynamic LLM token generation loop** |
| **Wave Matrix (WMMA)** | Agility 1.720 | Shader Model 6.8 | `WaveMatrix<float16_t, 16, 16>` | **+40-47% FP16 GEMM & Attention** acceleration on RDNA4 |
| **Enhanced Barriers** | Agility 1.606+ | Shader Model 6.7 | `D3D12_BARRIER_GROUP` | **Eliminates pipeline stalls** between compute queue dispatches |
| **Bindless Descriptor Heap**| Agility 1.4+ | Shader Model 6.6 | `ResourceDescriptorHeap[index]` | Direct pointer arithmetic without descriptor table re-binding |
| **Wave Multi-Prefix** | Agility 1.4+ | Shader Model 6.6 | `WaveMultiPrefixSum()` | Fast intra-wave reduction for RMSNorm and Softmax |
| **Float UAV Atomics** | Agility 1.4+ | Shader Model 6.6 | `InterlockedAdd(float)` | Lock-free parallel scatter-add for MoE token routing |
| **DirectStorage 1.4** | DirectStorage 1.4 | D3D12 API | `IDStorageFactory` / `IDStorageQueue` | **7 GB/s zero-copy PCIe Gen4 DMA** NVMe weight streaming |

---

## 2. Work Graphs 1.1 (Shader Model 6.8 / 6.9) Architecture

Work Graphs enable dynamic GPU-side node dispatch, eliminating CPU kernel launch overhead during LLM autoregressive token decoding:

```cpp
// Creating Work Graph State Object in D3D12 Agility SDK
D3D12_STATE_OBJECT_DESC state_obj_desc{};
state_obj_desc.Type = D3D12_STATE_OBJECT_TYPE_EXECUTABLE;

// Define Work Graph Program Node
D3D12_WORK_GRAPH_DESC wg_desc{};
wg_desc.ProgramName = L"LLM_Token_Generation_Graph";
wg_desc.Flags = D3D12_WORK_GRAPH_FLAG_ADD_TO_EXISTING_WORK_GRAPH;
```

HLSL Work Graph Node:

```hlsl
// HLSL Work Graph Token Generator Node
[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1, 1, 1)]
void TokenDecoderNode(
    uint3 dispatch_id : SV_DispatchThreadID,
    DispatchNodeInputRecord<TokenInput> input_record,
    NodeOutput<NextTokenRecord> output_node
) {
    // Process Token embedding & attention
    // Output next token to downstream compute node on GPU
    ThreadNodeOutputRecords<NextTokenRecord> out_rec = output_node.GetThreadNodeOutputRecords(1);
    out_rec[0].token_id = input_record.token_id + 1;
    out_rec.OutputComplete();
}
```

---

## 3. Shader Model 6.7 Enhanced Barriers

Enhanced Barriers allow precise memory and layout transitions without flushing the GPU pipeline:

```cpp
void RecordEnhancedBarrier(ID3D12GraphicsCommandList7* cmd_list, ID3D12Resource* buffer) {
    D3D12_BUFFER_BARRIER barrier{};
    barrier.SyncBefore = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
    barrier.SyncAfter = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
    barrier.AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
    barrier.AccessAfter = D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
    barrier.pResource = buffer;

    D3D12_BARRIER_GROUP group{};
    group.Type = D3D12_BARRIER_TYPE_BUFFER;
    group.NumBarriers = 1;
    group.pBufferBarriers = &barrier;

    cmd_list->Barrier(1, &group);
}
```
