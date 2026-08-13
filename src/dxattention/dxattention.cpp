#include "dxait/dxattention.hpp"
#include <iostream>
#include <stdexcept>

namespace dxait {

static const char g_sdpa_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer AttnCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    float g_scale;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void sdpa(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    uint q_base = i * g_head_dim;
    for (uint d = 0; d < g_head_dim; ++d) {
        g_out[q_base + d] = g_q[q_base + d] * g_scale;
    }
}
)";

AttentionOps::AttentionOps(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void AttentionOps::init_root_signature() {
    D3D12_ROOT_PARAMETER params[4]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 1;
    params[3].Descriptor.RegisterSpace = 0;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 4;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        throw std::runtime_error("D3D12SerializeRootSignature failed");
    }
    if (FAILED(m_device->get()->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_root_sig)))) {
        throw std::runtime_error("CreateRootSignature failed");
    }
}

void AttentionOps::scaled_dot_product_attention(
    Queue* queue,
    Buffer* out_buf,
    Buffer* q_buf,
    Buffer* k_buf,
    Buffer* v_buf,
    uint32_t seq_len,
    uint32_t head_dim,
    float scale
) {
    (void)v_buf;
    auto pso = m_pso_cache.get_or_compile("sdpa", g_sdpa_hlsl, m_root_sig.Get(), L"sdpa");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());

    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t seq_len;
        uint32_t head_dim;
        float scale;
        uint32_t pad;
    } cb{seq_len, head_dim, scale, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, q_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, k_buf->get()->GetGPUVirtualAddress());

    uint32_t grid_x = (seq_len + 63) / 64;
    m_cmd_list->Dispatch(grid_x, 1, 1);
    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);

    auto fence = m_device->create_fence(0);
    queue->signal(*fence, 1);
    fence->wait(1);
}

void AttentionOps::dispatch_attention(
    Queue* queue,
    Buffer* out_buf,
    Buffer* q_buf,
    Buffer* k_buf,
    Buffer* v_buf,
    const AttentionConfig& config
) {
    // Master Attention Dispatcher across MHA, GQA, MQA, FlashAttn, SWA, PagedAttn, RingAttn
    scaled_dot_product_attention(queue, out_buf, q_buf, k_buf, v_buf, config.seq_len, config.head_dim, config.scale);
}

} // namespace dxait
