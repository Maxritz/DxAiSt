#include "dxait/dxcache.hpp"
#include "dxait/dxjit.hpp"
#include <stdexcept>

namespace dxait {

// Hadamard transform kernel. Applies the unnormalised Walsh-Hadamard transform
// to a power-of-two sized buffer of floats, in place. Used as a cheap orthogonal
// decorrelation pass over the KV cache before quantisation (KIVI-style idea).

static const char g_hadamard_hlsl[] = R"(
RWStructuredBuffer<float> g_data : register(u0);

cbuffer HadamardCB : register(b0) {
    uint g_n;
    uint g_len;    // current stage span
    uint g_pad[2];
};

[numthreads(64, 1, 1)]
void hadamard_kernel(uint3 id : SV_DispatchThreadID) {
    uint k = id.x;
    if (k >= g_n / 2) return;

    uint block = k / (g_len / 2);
    uint within = k % (g_len / 2);
    uint i0 = block * g_len + within;
    uint i1 = i0 + g_len / 2;

    float a = g_data[i0];
    float b = g_data[i1];
    g_data[i0] = a + b;
    g_data[i1] = a - b;
}
)";

AdvancedKVCache::AdvancedKVCache(Device* device, KVCacheType type, uint64_t max_bytes)
    : m_device(device), m_type(type) {
    m_cache_buffer = device->create_buffer(max_bytes, MemLocation::Default);
}

void AdvancedKVCache::apply_hadamard_transform(Queue* queue) {
    if (m_type != KVCacheType::HadamardTransform) return;
    if (!m_cache_buffer) return;

    // Work on the buffer as a power-of-two float count (clamped down).
    uint64_t float_count = m_cache_buffer->size() / sizeof(float);
    uint32_t n = 1;
    while ((uint64_t)(n << 1) <= float_count) n <<= 1;
    if (n < 2) return;

    ShaderCompiler compiler;
    PipelineCache pso_cache(m_device->get());

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumParameters = 2;
    rs_desc.pParameters = params;
    rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    auto root_sig = compiler.create_root_signature(m_device->get(), rs_desc);

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)))) return;
    if (FAILED(m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) return;
    list->Close();

    auto fence = m_device->create_fence(0);
    uint64_t fence_val = 0;

    for (uint32_t len = 2; len <= n; len <<= 1) {
        auto pso = pso_cache.get_or_compile("hadamard_stage", g_hadamard_hlsl, root_sig.Get(), L"hadamard_kernel");

        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(root_sig.Get());

        struct CB { uint32_t n, len, pad[2]; } cb{n, len, {0, 0}};
        list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
        list->SetComputeRootUnorderedAccessView(1, m_cache_buffer->get()->GetGPUVirtualAddress());

        list->Dispatch((n / 2 + 63) / 64, 1, 1);
        list->Close();

        ID3D12CommandList* lists[] = { list.Get() };
        queue->execute(lists, 1);
        queue->signal(*fence, ++fence_val);
        fence->wait(fence_val);
    }
}

} // namespace dxait
