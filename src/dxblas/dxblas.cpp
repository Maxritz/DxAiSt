#include "dxait/dxblas.hpp"

namespace dxait {

static const char g_vec_add_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in1 : register(t0);
StructuredBuffer<float> g_in2 : register(t1);

cbuffer ElementwiseCB : register(b0) {
    uint g_count;
    float g_alpha;
    float g_beta;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void vec_add(uint3 id : SV_DispatchThreadID) {
    if (id.x < g_count) {
        g_out[id.x] = g_alpha * g_in1[id.x] + g_beta * g_in2[id.x];
    }
}
)";

static const char g_gemm_tiled_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_a : register(t0);
StructuredBuffer<float> g_b : register(t1);

cbuffer GemmCB : register(b0) {
    uint g_M;
    uint g_N;
    uint g_K;
    uint g_pad;
};

#define TILE 16
groupshared float s_a[TILE][TILE];
groupshared float s_b[TILE][TILE];

[numthreads(TILE, TILE, 1)]
void gemm_tiled(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint ltid : SV_GroupIndex) {
    uint tx = ltid % TILE;
    uint ty = ltid / TILE;
    uint row = gid.y * TILE + ty;
    uint col = gid.x * TILE + tx;
    if (row >= g_M || col >= g_N) return;

    float acc = 0.0f;
    for (uint kk = 0; kk < g_K; kk += TILE) {
        // cooperative load A and B tiles into shared memory
        uint ak = kk + tx; // for A: k index
        uint bk = kk + ty; // for B: k index
        s_a[ty][tx] = (row < g_M && ak < g_K) ? g_a[row * g_K + ak] : 0.0f;
        s_b[ty][tx] = (bk < g_K && col < g_N) ? g_b[bk * g_N + col] : 0.0f;
        GroupMemoryBarrierWithGroupSync();

        // compute inner product over this k-tile
        for (uint k = 0; k < TILE; ++k) {
            acc += s_a[ty][k] * s_b[k][tx];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    g_out[row * g_N + col] = acc;
}
)";

// F16 packed GEMM using dot2add: A and B are F16, K must be even. Loads two
// halves per uint, accumulates with dot2add (v_dot2_f32_f16 on RDNA2, 2 FMA in
// 1 instruction). Mirrors llama mms_f16.
static const char g_gemm_f16_dot2_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<uint> g_a : register(t0);
StructuredBuffer<uint> g_b : register(t1);

cbuffer GemmF16CB : register(b0) {
    uint g_M;
    uint g_N;
    uint g_K;      // even
    uint g_pad;
};

[numthreads(16, 16, 1)]
void gemm_f16_dot2(uint3 id : SV_DispatchThreadID) {
    uint row = id.y;
    uint col = id.x;
    if (row >= g_M || col >= g_N) return;

    float acc = 0.0f;
    // K halves packed per uint: 2 elements per load
    for (uint k = 0; k + 1u < g_K; k += 2u) {
        uint aw = g_a[row * (g_K / 2u) + k / 2u];
        uint bw = g_b[k * (g_N / 2u) + col / 2u];
        half2 ha = half2(f16tof32(aw & 0xFFFFu), f16tof32(aw >> 16));
        half2 hb = half2(f16tof32(bw & 0xFFFFu), f16tof32(bw >> 16));
        // dot2add on RDNA2/RDNA4 (v_dot2_f32_f16). Fallback manual if compiler lacks it.
        acc = dot2add(ha, hb, acc);
    }
    if ((g_K & 1u) != 0u) {
        uint k = g_K - 1u;
        uint aw = g_a[row * (g_K / 2u) + k / 2u];
        uint bw = g_b[k * (g_N / 2u) + col / 2u];
        float a = f16tof32((k & 1u) ? (aw >> 16) : (aw & 0xFFFFu));
        float b = f16tof32((col & 1u) ? (bw >> 16) : (bw & 0xFFFFu));
        acc += a * b;
    }
    g_out[row * g_N + col] = acc;
}
)";

BLAS::BLAS(Device* device)
    : m_device(device), m_pso_cache(device->get()), m_fence(device->create_fence(0)) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("Failed to create compute command allocator");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("Failed to create compute command list");
    }
    m_cmd_list->Close();
}

void BLAS::init_root_signature() {
    D3D12_ROOT_PARAMETER params[4]{};

    // Root parameter 0: Constant Buffer (CBV)
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Root parameter 1: UAV (Out Buffer)
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Root parameter 2: SRV (In1 Buffer)
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Root parameter 3: SRV (In2 Buffer)
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

void BLAS::vec_add(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in1_buf,
    Buffer* in2_buf,
    uint32_t count,
    float alpha,
    float beta
) {
    auto pso = m_pso_cache.get_or_compile("vec_add", g_vec_add_hlsl, m_root_sig.Get(), L"vec_add");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());

    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t count;
        float alpha;
        float beta;
        uint32_t pad;
    } cb{count, alpha, beta, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in1_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in2_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (count + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

void BLAS::gemm(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in_a,
    Buffer* in_b,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    auto pso = m_pso_cache.get_or_compile("gemm_tiled", g_gemm_tiled_hlsl, m_root_sig.Get(), L"gemm_tiled");

    if (m_fence_val > 0) m_fence->wait(m_fence_val);
    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t M, N, K, pad;
    } cb{M, N, K, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_a->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in_b->get()->GetGPUVirtualAddress());

    uint32_t grid_x = (N + 15) / 16;
    uint32_t grid_y = (M + 15) / 16;
    m_cmd_list->Dispatch(grid_x, grid_y, 1);

    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
    queue->signal(*m_fence, ++m_fence_val);
}

void BLAS::gemm_f16_dot2(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in_a_f16,
    Buffer* in_b_f16,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    auto pso = m_pso_cache.get_or_compile("gemm_f16_dot2", g_gemm_f16_dot2_hlsl, m_root_sig.Get(), L"gemm_f16_dot2");

    if (m_fence_val > 0) m_fence->wait(m_fence_val);
    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t M, N, K, pad;
    } cb{M, N, K, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_a_f16->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in_b_f16->get()->GetGPUVirtualAddress());

    m_cmd_list->Dispatch((N + 15) / 16, (M + 15) / 16, 1);
    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
    queue->signal(*m_fence, ++m_fence_val);
}

} // namespace dxait
