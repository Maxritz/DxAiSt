#include "dxait/dxfft.hpp"
#include <stdexcept>

namespace dxait {

// Iterative radix-2 Cooley-Tukey FFT.
// Two kernels:
//   1. bit_reverse: permute input into bit-reversed order.
//   2. fft_stage:   one butterfly stage per dispatch (log2(n) dispatches).
// Treated as complex pairs (re, im) in separate buffers. n must be power of two.

static const char g_bit_reverse_hlsl[] = R"(
RWStructuredBuffer<float> g_out_r : register(u0);
RWStructuredBuffer<float> g_out_i : register(u1);
StructuredBuffer<float> g_in_r : register(t0);
StructuredBuffer<float> g_in_i : register(t1);

cbuffer FFTBitRevCB : register(b0) {
    uint g_n;
    uint g_log2n;
    uint2 g_pad;
};

uint bit_reverse(uint x, uint bits) {
    uint r = 0;
    for (uint b = 0; b < bits; ++b) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

[numthreads(64, 1, 1)]
void bit_reverse_kernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g_n) return;
    uint rev = bit_reverse(id.x, g_log2n);
    g_out_r[rev] = g_in_r[id.x];
    g_out_i[rev] = g_in_i[id.x];
}
)";

static const char g_fft_stage_hlsl[] = R"(
RWStructuredBuffer<float> g_out_r : register(u0);
RWStructuredBuffer<float> g_out_i : register(u1);

cbuffer FFTSstageCB : register(b0) {
    uint g_n;
    uint g_len;    // current butterfly span (2 per stage doubling)
    uint g_stage;  // 1-based stage index
    uint g_pad;
};

[numthreads(64, 1, 1)]
void fft_stage_kernel(uint3 id : SV_DispatchThreadID) {
    uint k = id.x;
    if (k >= g_n / 2) return;

    // k-th butterfly: pair (k) and (k + g_len/2) within each g_len block
    uint block = k / (g_len / 2);
    uint within = k % (g_len / 2);
    uint i0 = block * g_len + within;
    uint i1 = i0 + g_len / 2;

    float angle = -2.0f * 3.14159265358979323846f * (float)within / (float)g_len;
    float cw = cos(angle);
    float sw = sin(angle);

    float ar = g_out_r[i0];
    float ai = g_out_i[i0];
    float br = g_out_r[i1];
    float bi = g_out_i[i1];

    float tr = br * cw - bi * sw;
    float ti = br * sw + bi * cw;

    g_out_r[i0] = ar + tr;
    g_out_i[i0] = ai + ti;
    g_out_r[i1] = ar - tr;
    g_out_i[i1] = ai - ti;
}
)";

FFTOps::FFTOps(Device* device)
    : m_device(device), m_pso_cache(device->get()), m_fence(device->create_fence(0)) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void FFTOps::init_root_signature() {
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

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 1;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 0;
    params[3].Descriptor.RegisterSpace = 0;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[4].Descriptor.ShaderRegister = 1;
    params[4].Descriptor.RegisterSpace = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 5;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void FFTOps::fft_1d_radix2(
    Queue* queue,
    Buffer* out_real,
    Buffer* out_imag,
    Buffer* in_real,
    Buffer* in_imag,
    uint32_t n
) {
    if (n < 2 || (n & (n - 1)) != 0) return; // must be power of two >= 2

    uint32_t log2n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) ++log2n;

    // 1. Bit-reversal permutation (in -> out)
    {
        auto pso = m_pso_cache.get_or_compile("fft_bit_reverse", g_bit_reverse_hlsl, m_root_sig.Get(), L"bit_reverse_kernel");

        if (m_fence_val > 0) m_fence->wait(m_fence_val);
        m_cmd_alloc->Reset();
        m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
        m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

        struct CB { uint32_t n, log2n, pad[2]; } cb{n, log2n, {0, 0}};
        m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
        m_cmd_list->SetComputeRootUnorderedAccessView(1, out_real->get()->GetGPUVirtualAddress());
        m_cmd_list->SetComputeRootUnorderedAccessView(2, out_imag->get()->GetGPUVirtualAddress());
        m_cmd_list->SetComputeRootShaderResourceView(3, in_real->get()->GetGPUVirtualAddress());
        m_cmd_list->SetComputeRootShaderResourceView(4, in_imag->get()->GetGPUVirtualAddress());

        m_cmd_list->Dispatch((n + 63) / 64, 1, 1);
        m_cmd_list->Close();
        ID3D12CommandList* lists[] = { m_cmd_list.Get() };
        queue->execute(lists, 1);
        queue->signal(*m_fence, ++m_fence_val);
    }

    // 2. log2(n) butterfly stages, in place on out buffers
    for (uint32_t len = 2, stage = 1; len <= n; len <<= 1, ++stage) {
        auto pso = m_pso_cache.get_or_compile("fft_stage", g_fft_stage_hlsl, m_root_sig.Get(), L"fft_stage_kernel");

        if (m_fence_val > 0) m_fence->wait(m_fence_val);
        m_cmd_alloc->Reset();
        m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
        m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

        struct CB { uint32_t n, len, stage, pad; } cb{n, len, stage, 0};
        m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
        m_cmd_list->SetComputeRootUnorderedAccessView(1, out_real->get()->GetGPUVirtualAddress());
        m_cmd_list->SetComputeRootUnorderedAccessView(2, out_imag->get()->GetGPUVirtualAddress());

        m_cmd_list->Dispatch((n / 2 + 63) / 64, 1, 1);
        m_cmd_list->Close();
        ID3D12CommandList* lists[] = { m_cmd_list.Get() };
        queue->execute(lists, 1);
        queue->signal(*m_fence, ++m_fence_val);
    }
}

} // namespace dxait
