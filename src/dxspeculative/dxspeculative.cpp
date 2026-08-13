#include "dxait/dxspeculative.hpp"

namespace dxait {

SpeculativeEngine::SpeculativeEngine(Device* device) : m_device(device), m_pso_cache(device->get()), m_fence(device->create_fence(0)) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void SpeculativeEngine::init_root_signature() {
    D3D12_ROOT_PARAMETER params[5]{};

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

    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[4].Descriptor.ShaderRegister = 2;
    params[4].Descriptor.RegisterSpace = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 5;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void SpeculativeEngine::verify_draft_tokens(
    Queue* queue,
    Buffer* accept_mask_buf,
    Buffer* target_probs_buf,
    Buffer* draft_probs_buf,
    Buffer* draft_tokens_buf,
    uint32_t num_draft_tokens,
    uint32_t vocab_size,
    float random_val
) {
    const char hlsl_src[] = R"(
    RWStructuredBuffer<uint> g_accept_mask : register(u0);
    StructuredBuffer<float> g_target_probs : register(t0);
    StructuredBuffer<float> g_draft_probs : register(t1);
    StructuredBuffer<uint> g_draft_tokens : register(t2);

    cbuffer SpeculativeCB : register(b0) {
        uint g_num_draft_tokens;
        uint g_vocab_size;
        float g_random_val;
        uint g_pad;
    };

    [numthreads(32, 1, 1)]
    void speculative_verify_kernel(uint3 id : SV_DispatchThreadID) {
        uint token_idx = id.x;
        if (token_idx >= g_num_draft_tokens) return;

        // Speculative acceptance rule (Leviathan et al.):
        // accept draft token t if r <= min(1, p_target(t) / p_draft(t))
        uint tok = g_draft_tokens[token_idx];
        float p_target = tok < g_vocab_size ? g_target_probs[tok] : 0.0f;
        float p_draft  = tok < g_vocab_size ? g_draft_probs[tok]  : 0.0f;
        float ratio = p_draft > 1e-12f ? (p_target / p_draft) : 0.0f;
        float r = g_random_val;
        g_accept_mask[token_idx] = (r <= ratio) ? 1u : 0u;
    }
    )";

    auto pso = m_pso_cache.get_or_compile("speculative_verify", hlsl_src, m_root_sig.Get(), L"speculative_verify_kernel");

    if (m_fence_val > 0) m_fence->wait(m_fence_val);
    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t num_draft_tokens;
        uint32_t vocab_size;
        float random_val;
        uint32_t pad;
    } cb{num_draft_tokens, vocab_size, random_val, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, accept_mask_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, target_probs_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, draft_probs_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(4, draft_tokens_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (num_draft_tokens + 31) / 32;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
    queue->signal(*m_fence, ++m_fence_val);
}

} // namespace dxait
