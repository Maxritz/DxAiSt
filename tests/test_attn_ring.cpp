#include "dxait/dxait.hpp"
#include "dxait/dxjit.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// RingAttention (single-GPU simulation): the sequence is split into ring chunks;
// each chunk is treated as if owned by one device in a ring. Attention proceeds
// by visiting KV chunks in ring order and accumulating with online softmax, the
// same reduction math a distributed ring would perform. This verifies the ring
// reduction logic on one device.

static const char g_ring_hlsl[] = R"(
#define RING_MAX_DIM 128
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer RingCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_ring_size;   // tokens per ring chunk
    uint g_pad0;
    float g_scale;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void ring_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    if (g_head_dim > RING_MAX_DIM) return;
    uint q_head = id.y;
    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    // visit KV chunks in ring order: chunk r holds keys [r*ring_size, (r+1)*ring_size)
    float m = -1e30f;
    float l = 0.0f;
    float acc[RING_MAX_DIM];
    for (uint d = 0; d < g_head_dim; ++d) acc[d] = 0.0f;

    for (uint r = 0; r * g_ring_size < g_seq_len; ++r) {
        uint j0 = r * g_ring_size;
        // find chunk max (causal: keys <= i)
        float c_max = -1e30f;
        for (uint j = j0; j < j0 + g_ring_size && j <= i && j < g_seq_len; ++j) {
            float dot = 0.0f;
            uint k_base = (q_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d) dot += g_q[q_base + d] * g_k[k_base + d];
            c_max = max(c_max, dot * g_scale);
        }

        float new_m = max(m, c_max);
        float rescale = exp(m - new_m);
        float tmp[RING_MAX_DIM];
        for (uint d = 0; d < g_head_dim; ++d) tmp[d] = 0.0f;
        float c_l = 0.0f;

        for (uint j = j0; j < j0 + g_ring_size && j <= i && j < g_seq_len; ++j) {
            float dot = 0.0f;
            uint k_base = (q_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d) dot += g_q[q_base + d] * g_k[k_base + d];
            float p = exp((dot * g_scale) - new_m);
            c_l += p;
            uint v_base = (q_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d) tmp[d] += p * g_v[v_base + d];
        }

        for (uint d = 0; d < g_head_dim; ++d) acc[d] = acc[d] * rescale + tmp[d];
        l = l * rescale + c_l;
        m = new_m;
    }

    for (uint d = 0; d < g_head_dim; ++d)
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = (l > 0.0f) ? (acc[d] / l) : 0.0f;
}
)";

int main() {
    printf("DXAiT RingAttention (single-GPU) Test\n");
    printf("=====================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t seq = 8;
    constexpr uint32_t dim = 4;
    constexpr uint32_t n_qh = 1;
    constexpr uint32_t ring_size = 3;
    constexpr float scale = 0.5f;

    constexpr uint64_t bytes = (uint64_t)n_qh * seq * dim * sizeof(float);

    std::vector<float> q(n_qh * seq * dim), k(n_qh * seq * dim), v(n_qh * seq * dim);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = ((float)i * 0.13f) - 1.0f;
    for (uint32_t i = 0; i < k.size(); ++i) k[i] = ((float)i * 0.07f) + 0.5f;
    for (uint32_t i = 0; i < v.size(); ++i) v[i] = ((float)i * 0.03f) + 0.1f;

    auto q_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto k_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto v_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto out_buf = device->create_buffer(bytes, dxait::MemLocation::Readback);
    std::memcpy(q_buf->map(), q.data(), bytes); q_buf->unmap();
    std::memcpy(k_buf->map(), k.data(), bytes); k_buf->unmap();
    std::memcpy(v_buf->map(), v.data(), bytes); v_buf->unmap();

    dxait::ShaderCompiler compiler;
    D3D12_ROOT_PARAMETER params[5]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0; params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 8; params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t p = 1; p <= 4; ++p) {
        params[p].ParameterType = (p == 1) ? D3D12_ROOT_PARAMETER_TYPE_UAV : D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[p].Descriptor.ShaderRegister = (p == 1) ? 0 : (p - 2);
        params[p].Descriptor.RegisterSpace = 0;
        params[p].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 5; rsd.pParameters = params; rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    auto root_sig = compiler.create_root_signature(device->get(), rsd);

    dxait::PipelineCache cache(device->get());
    auto pso = cache.get_or_compile("ring_attn", g_ring_hlsl, root_sig.Get(), L"ring_attn");

    auto cq = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);
    dxait::ComPtr<ID3D12CommandAllocator> alloc;
    dxait::ComPtr<ID3D12GraphicsCommandList> list;
    device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), pso.Get(), IID_PPV_ARGS(&list));
    list->SetComputeRootSignature(root_sig.Get());

    struct CB { uint32_t seq, dim, ring, pad0; float scale; uint32_t pad[3]; }
        cb{seq, dim, ring_size, 0, scale, {0, 0, 0}};
    list->SetComputeRoot32BitConstants(0, 8, &cb, 0);
    list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, q_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(3, k_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(4, v_buf->get()->GetGPUVirtualAddress());
    list->Dispatch((seq + 63) / 64, n_qh, 1);
    list->Close();
    ID3D12CommandList* ls[] = { list.Get() };
    cq->execute(ls, 1);
    cq->signal(*fence, 1);
    fence->wait(1);

    // CPU ref = plain causal softmax attention (ring reduction must match exactly)
    float* o = (float*)out_buf->map();
    bool ok = true;
    float max_err = 0.0f;
    for (uint32_t h = 0; h < n_qh; ++h) {
        for (uint32_t i = 0; i < seq; ++i) {
            float max_s = -1e30f;
            for (uint32_t j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dot += q[(h*seq+i)*dim+d] * k[(h*seq+j)*dim+d];
                max_s = (std::max)(max_s, dot * scale);
            }
            float sum = 0.0f;
            std::vector<float> w(seq, 0.0f);
            for (uint32_t j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dot += q[(h*seq+i)*dim+d] * k[(h*seq+j)*dim+d];
                w[j] = std::exp(dot * scale - max_s);
                sum += w[j];
            }
            float ref = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) acc += (w[j]/sum) * v[(h*seq+j)*dim+d];
                if (d == 0) ref = acc;
            }
            float got = o[(h*seq+i)*dim];
            float err = std::fabs(got - ref);
            if (err > max_err) max_err = err;
            if (err > 1e-2f) { printf("  mismatch i=%u got=%.5f ref=%.5f\n", i, got, ref); ok = false; }
        }
    }
    out_buf->unmap();

    printf("  max_err=%.5f %s\n", max_err, ok ? "PASS" : "FAIL");
    printf("\nResult: %s\n", ok ? "RingAttention PASSED" : "FAILED");
    return ok ? 0 : 1;
}
