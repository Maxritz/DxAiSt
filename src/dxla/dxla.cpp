#include "dxait/dxla.hpp"

#include <cstdio>
#include <stdexcept>

namespace dxait {

static const char g_elementwise_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_a : register(t0);
StructuredBuffer<float> g_b : register(t1);

cbuffer CB : register(b0) {
    uint g_count;
    int  g_op;
    float g_alpha;
    float g_beta;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g_count) return;
    float x = g_a[id.x];
    float y = g_b[id.x];
    float r = 0.0f;
    if (g_op == 0) r = x + y;
    else if (g_op == 1) r = x - y;
    else if (g_op == 2) r = x * y;
    else r = x / y;
    g_out[id.x] = g_alpha * r + g_beta;
}
)";

// erf via Abramowitz-Stegun 7.1.26 (max abs error ~1.5e-7).
static const char g_activation_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);

cbuffer CB : register(b0) {
    uint g_count;
    int  g_act;
    float g_alpha;
    uint g_pad;
};

float erf_approx(float x) {
    float sgn = x < 0.0f ? -1.0f : 1.0f;
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return sgn * y;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g_count) return;
    float x = g_in[id.x];
    float r = 0.0f;
    if (g_act == 0) r = x > 0.0f ? x : 0.0f;
    else if (g_act == 1) r = x * 0.5f * (1.0f + erf_approx(x * 0.70710678f));
    else if (g_act == 2) r = x / (1.0f + exp(-x));
    else if (g_act == 3) r = tanh(x);
    else if (g_act == 4) r = 1.0f / (1.0f + exp(-x));
    else r = x >= 0.0f ? x : g_alpha * x;
    g_out[id.x] = r;
}
)";

static const char g_rmsnorm_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);
StructuredBuffer<float> g_gamma : register(t1);

cbuffer CB : register(b0) {
    uint g_rows;
    uint g_dim;
    float g_eps;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_rows * g_dim) return;
    uint row = idx / g_dim;
    uint col = idx % g_dim;
    float s = 0.0f;
    for (uint j = 0; j < g_dim; ++j) { float v = g_in[row * g_dim + j]; s += v * v; }
    float rstd = rsqrt(s / (float)g_dim + g_eps);
    g_out[idx] = g_gamma[col] * g_in[idx] * rstd;
}
)";

static const char g_softmax_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);

cbuffer CB : register(b0) {
    uint g_rows;
    uint g_dim;
    uint g_pad0;
    uint g_pad1;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_rows * g_dim) return;
    uint row = idx / g_dim;
    uint col = idx % g_dim;
    float m = -3.4e38f;
    for (uint j = 0; j < g_dim; ++j) m = max(m, g_in[row * g_dim + j]);
    float s = 0.0f;
    for (uint j = 0; j < g_dim; ++j) s += exp(g_in[row * g_dim + j] - m);
    g_out[idx] = exp(g_in[idx] - m) / s;
}
)";

static const char g_reduce_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);

cbuffer CB : register(b0) {
    uint g_rows;
    uint g_dim;
    int  g_op;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= g_rows) return;
    float acc = 0.0f;
    if (g_op == 1) acc = -3.4e38f;
    else if (g_op == 2) acc = 3.4e38f;
    for (uint j = 0; j < g_dim; ++j) {
        float x = g_in[row * g_dim + j];
        if (g_op == 0) acc += x;
        else if (g_op == 1) acc = max(acc, x);
        else if (g_op == 2) acc = min(acc, x);
        else acc += x;
    }
    if (g_op == 3) acc /= (float)g_dim;
    g_out[row] = acc;
}
)";

// F16 GEMM, K even, A/B as uint32 little-endian half pairs, out F32 row-major.
// v_dot2_f32_f16 (dot2add) path, same as BLAS gemm_f16_dot2.
static const char g_gemm_f16_dot2_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<uint> g_a : register(t0);
StructuredBuffer<uint> g_b : register(t1);

cbuffer CB : register(b0) {
    uint g_M;
    uint g_N;
    uint g_K;
    uint g_pad;
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.y;
    uint col = id.x;
    if (row >= g_M || col >= g_N) return;
    float acc = 0.0f;
    for (uint k = 0; k + 1u < g_K; k += 2u) {
        uint aw = g_a[row * (g_K / 2u) + k / 2u];
        uint bw = g_b[col * (g_K / 2u) + k / 2u];
        half2 ha = half2(f16tof32(aw & 0xFFFFu), f16tof32(aw >> 16));
        half2 hb = half2(f16tof32(bw & 0xFFFFu), f16tof32(bw >> 16));
        acc = dot2add(ha, hb, acc);
    }
    g_out[row * g_N + col] = acc;
}
)";

// WMMA in HLSL requires SM6.9/6.10 LinAlg (preview driver + Agility SDK) -
// the SM6.8 WaveMatrix intrinsics were shelved and removed from DXC (PR #6807,
// released 1.8.2502). gemm_f16_wmma intentionally dispatches the dot2 kernel,
// which produces the same result on all hardware incl. RDNA4 WMMA hardware.
static const char g_gemm_f16_wmma_hlsl[] = R"()";

LA::LA(Device* device)
    : m_device(device), m_pso_cache(device->get()), m_fence(device->create_fence(0)) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("LA: failed to create compute command allocator");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("LA: failed to create compute command list");
    }
    m_cmd_list->Close();
}

void LA::init_root_signature() {
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

void LA::ensure_idle() {
    if (m_fence_val > 0) m_fence->wait(m_fence_val);
    m_cmd_alloc->Reset();
}

void LA::elementwise(Queue* q, Buffer* out, Buffer* in0, Buffer* in1, uint32_t count, LAOp op, float alpha, float beta) {
    auto pso = m_pso_cache.get_or_compile("la_elementwise", g_elementwise_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t count; int32_t op; float alpha; float beta; } cb{count, (int32_t)op, alpha, beta};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in0->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in1->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch((count + 63) / 64, 1, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::activation(Queue* q, Buffer* out, Buffer* in, uint32_t count, LAActivation act, float alpha) {
    auto pso = m_pso_cache.get_or_compile("la_activation", g_activation_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t count; int32_t act; float alpha; uint32_t pad; } cb{count, (int32_t)act, alpha, 0};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch((count + 63) / 64, 1, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::rmsnorm(Queue* q, Buffer* out, Buffer* in, Buffer* gamma, uint32_t rows, uint32_t dim, float eps) {
    auto pso = m_pso_cache.get_or_compile("la_rmsnorm", g_rmsnorm_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t rows; uint32_t dim; float eps; uint32_t pad; } cb{rows, dim, eps, 0};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, gamma->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch(((uint64_t)rows * dim + 63) / 64, 1, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::softmax(Queue* q, Buffer* out, Buffer* in, uint32_t rows, uint32_t dim) {
    auto pso = m_pso_cache.get_or_compile("la_softmax", g_softmax_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t rows; uint32_t dim; uint32_t pad0; uint32_t pad1; } cb{rows, dim, 0, 0};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch(((uint64_t)rows * dim + 63) / 64, 1, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::reduce(Queue* q, Buffer* out, Buffer* in, uint32_t rows, uint32_t dim, LAReduce op) {
    auto pso = m_pso_cache.get_or_compile("la_reduce", g_reduce_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t rows; uint32_t dim; int32_t op; uint32_t pad; } cb{rows, dim, (int32_t)op, 0};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch((rows + 63) / 64, 1, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::gemm_f16_dot2(Queue* q, Buffer* out_f32, Buffer* a_f16, Buffer* b_f16, uint32_t M, uint32_t N, uint32_t K) {
    auto pso = m_pso_cache.get_or_compile("la_gemm_f16_dot2", g_gemm_f16_dot2_hlsl, m_root_sig.Get(), L"main");
    ensure_idle();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());
    struct CB { uint32_t M, N, K, pad; } cb{M, N, K, 0};
    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_f32->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, a_f16->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, b_f16->get()->GetGPUVirtualAddress());
    m_cmd_list->Dispatch((N + 15) / 16, (M + 15) / 16, 1);
    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    q->execute(lists, 1);
    q->signal(*m_fence, ++m_fence_val);
}

void LA::gemm_f16_wmma(Queue* q, Buffer* out_f32, Buffer* a_f16, Buffer* b_f16, uint32_t M, uint32_t N, uint32_t K) {
    // No released HLSL compiler emits WaveMatrix ops (SM6.8 type shelved;
    // SM6.9/6.10 LinAlg needs preview driver + Agility SDK). dot2 gives the
    // same result on all hardware. Kept as a distinct entry so a future WMMA
    // path can slot in without changing callers.
    m_wmma_status = "gemm_f16_wmma: WMMA requires SM6.9/6.10 LinAlg (preview driver + Agility SDK) - excluded; using dot2";
    std::fprintf(stderr, "LA: %s\n", m_wmma_status.c_str());
    gemm_f16_dot2(q, out_f32, a_f16, b_f16, M, N, K);
}

} // namespace dxait