#include "dxait/dxmath.hpp"

namespace dxait {

static const char g_rms_norm_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);
StructuredBuffer<float> g_weight : register(t1);

cbuffer NormCB : register(b0) {
    uint g_row_dim;
    float g_eps;
    uint2 g_pad;
};

[numthreads(64, 1, 1)]
void rms_norm(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint base = row * g_row_dim;
    float ss = 0.0f;
    for (uint i = 0; i < g_row_dim; ++i) {
        float v = g_in[base + i];
        ss += v * v;
    }
    float scale = rsqrt(ss / (float)g_row_dim + g_eps);
    for (uint i = 0; i < g_row_dim; ++i) {
        g_out[base + i] = g_in[base + i] * scale * g_weight[i];
    }
}
)";

static const char g_softmax_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);

cbuffer SoftmaxCB : register(b0) {
    uint g_row_dim;
    float g_temp;
    uint2 g_pad;
};

[numthreads(64, 1, 1)]
void softmax(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint base = row * g_row_dim;
    float inv_temp = 1.0f / max(g_temp, 1e-6f);
    float max_val = -1e30f;
    for (uint i = 0; i < g_row_dim; ++i) {
        max_val = max(max_val, g_in[base + i] * inv_temp);
    }
    float sum = 0.0f;
    for (uint i = 0; i < g_row_dim; ++i) {
        float ex = exp((g_in[base + i] * inv_temp) - max_val);
        g_out[base + i] = ex;
        sum += ex;
    }
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < g_row_dim; ++i) {
        g_out[base + i] *= inv_sum;
    }
}
)";

static const char g_rope_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);

cbuffer RopeCB : register(b0) {
    uint g_row_dim;
    uint g_head_dim;
    uint g_pos;
    float g_theta;
};

[numthreads(64, 1, 1)]
void rope(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint base = row * g_row_dim;
    for (uint i = 0; i < g_head_dim; i += 2) {
        float freq = pow(g_theta, -(float)i / (float)g_head_dim);
        float arg = (float)g_pos * freq;
        float cos_a = cos(arg);
        float sin_a = sin(arg);
        float v0 = g_in[base + i];
        float v1 = g_in[base + i + 1];
        g_out[base + i] = v0 * cos_a - v1 * sin_a;
        g_out[base + i + 1] = v0 * sin_a + v1 * cos_a;
    }
}
)";

MathOps::MathOps(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void MathOps::init_root_signature() {
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

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void MathOps::rms_norm(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in_buf,
    Buffer* weight_buf,
    uint32_t num_rows,
    uint32_t row_dim,
    float eps
) {
    auto pso = m_pso_cache.get_or_compile("rms_norm", g_rms_norm_hlsl, m_root_sig.Get(), L"rms_norm");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t row_dim;
        float eps;
        uint32_t pad[2];
    } cb{row_dim, eps, {0, 0}};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, weight_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (num_rows + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

void MathOps::softmax(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in_buf,
    uint32_t num_rows,
    uint32_t row_dim
) {
    auto pso = m_pso_cache.get_or_compile("softmax", g_softmax_hlsl, m_root_sig.Get(), L"softmax");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t row_dim;
        uint32_t pad[3];
    } cb{row_dim, {0, 0, 0}};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (num_rows + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
