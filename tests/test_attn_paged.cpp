#include "dxait/dxait.hpp"
#include "dxait/dxjit.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// PagedAttention: KV stored in fixed-size blocks (page_size tokens), a block
// table maps logical block index to physical block. Kernel gathers KV via the
// table. Standalone test: builds own root sig + PSO, dispatches, CPU-verifies.

static const char g_paged_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);
StructuredBuffer<uint> g_table : register(t3);

cbuffer PagedCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_page_size;
    uint g_n_blocks;
    float g_scale;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void paged_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    uint q_head = id.y;

    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    float max_s = -1e30f;
    for (uint j = 0; j < g_seq_len; ++j) {
        uint logical_block = j / g_page_size;
        uint in_block = j % g_page_size;
        uint phys = g_table[q_head * g_n_blocks + logical_block];
        float dot = 0.0f;
        uint k_base = ((q_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d)
            dot += g_q[q_base + d] * g_k[k_base + d];
        max_s = max(max_s, dot * g_scale);
    }

    float sum = 0.0f;
    float w[256];
    for (uint j = 0; j < g_seq_len; ++j) {
        uint logical_block = j / g_page_size;
        uint in_block = j % g_page_size;
        uint phys = g_table[q_head * g_n_blocks + logical_block];
        float dot = 0.0f;
        uint k_base = ((q_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d)
            dot += g_q[q_base + d] * g_k[k_base + d];
        w[j] = exp((dot * g_scale) - max_s);
        sum += w[j];
    }

    for (uint d = 0; d < g_head_dim; ++d) {
        float o = 0.0f;
        for (uint j = 0; j < g_seq_len; ++j) {
            uint logical_block = j / g_page_size;
            uint in_block = j % g_page_size;
            uint phys = g_table[q_head * g_n_blocks + logical_block];
            uint v_base = ((q_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
            o += (w[j] / sum) * g_v[v_base + d];
        }
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = o;
    }
}
)";

int main() {
    printf("DXAiT PagedAttention Block-Table Test\n");
    printf("=====================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t seq = 8;
    constexpr uint32_t dim = 4;
    constexpr uint32_t n_qh = 1;
    constexpr uint32_t page_size = 4;
    constexpr float scale = 0.5f;
    constexpr uint32_t n_blocks = (seq + page_size - 1) / page_size;

    constexpr uint64_t q_bytes = (uint64_t)n_qh * seq * dim * sizeof(float);
    constexpr uint64_t kv_bytes = (uint64_t)n_qh * n_blocks * page_size * dim * sizeof(float);
    constexpr uint64_t table_bytes = (uint64_t)n_qh * n_blocks * sizeof(uint32_t);

    std::vector<float> q(n_qh * seq * dim), k(n_qh * n_blocks * page_size * dim), v(n_qh * n_blocks * page_size * dim);
    std::vector<uint32_t> table(n_qh * n_blocks);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = ((float)i * 0.13f) - 1.0f;
    for (uint32_t i = 0; i < k.size(); ++i) k[i] = ((float)i * 0.07f) + 0.5f;
    for (uint32_t i = 0; i < v.size(); ++i) v[i] = ((float)i * 0.03f) + 0.1f;
    // Non-identity table: swap block 0 and block 1
    table[0] = 1; table[1] = 0;

    auto q_buf = device->create_buffer(q_bytes, dxait::MemLocation::Upload);
    auto k_buf = device->create_buffer(kv_bytes, dxait::MemLocation::Upload);
    auto v_buf = device->create_buffer(kv_bytes, dxait::MemLocation::Upload);
    auto table_buf = device->create_buffer(table_bytes, dxait::MemLocation::Upload);
    auto out_buf = device->create_buffer(q_bytes, dxait::MemLocation::Readback);
    std::memcpy(q_buf->map(), q.data(), q_bytes); q_buf->unmap();
    std::memcpy(k_buf->map(), k.data(), kv_bytes); k_buf->unmap();
    std::memcpy(v_buf->map(), v.data(), kv_bytes); v_buf->unmap();
    std::memcpy(table_buf->map(), table.data(), table_bytes); table_buf->unmap();

    // Root signature: constants + UAV + 4 SRVs (q,k,v,table)
    dxait::ShaderCompiler compiler;
    D3D12_ROOT_PARAMETER params[6]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0; params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 8; params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t p = 1; p <= 5; ++p) {
        params[p].ParameterType = (p == 1) ? D3D12_ROOT_PARAMETER_TYPE_UAV : D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[p].Descriptor.ShaderRegister = (p == 1) ? 0 : (p - 2);
        params[p].Descriptor.RegisterSpace = 0;
        params[p].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 6; rsd.pParameters = params; rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    auto root_sig = compiler.create_root_signature(device->get(), rsd);

    dxait::PipelineCache cache(device->get());
    auto pso = cache.get_or_compile("paged_attn", g_paged_hlsl, root_sig.Get(), L"paged_attn");

    auto cq = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);
    dxait::ComPtr<ID3D12CommandAllocator> alloc;
    dxait::ComPtr<ID3D12GraphicsCommandList> list;
    device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), pso.Get(), IID_PPV_ARGS(&list));
    list->SetComputeRootSignature(root_sig.Get());

    struct CB { uint32_t seq, dim, page, nblocks; float scale; uint32_t pad[3]; }
        cb{seq, dim, page_size, n_blocks, scale, {0, 0, 0}};
    list->SetComputeRoot32BitConstants(0, 8, &cb, 0);
    list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, q_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(3, k_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(4, v_buf->get()->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(5, table_buf->get()->GetGPUVirtualAddress());
    list->Dispatch((seq + 63) / 64, n_qh, 1);
    list->Close();
    ID3D12CommandList* ls[] = { list.Get() };
    cq->execute(ls, 1);
    cq->signal(*fence, 1);
    fence->wait(1);

    // CPU ref: with table {1,0}, logical block0 -> physical1, block1 -> physical0
    auto k_off = [&](uint32_t j) {
        uint32_t lb = j / page_size, ib = j % page_size;
        uint32_t phys = table[lb];
        return phys * page_size * dim + ib * dim;
    };
    auto v_off = [&](uint32_t j) {
        uint32_t lb = j / page_size, ib = j % page_size;
        uint32_t phys = table[lb];
        return phys * page_size * dim + ib * dim;
    };

    float* o = (float*)out_buf->map();
    bool ok = true;
    float max_err = 0.0f;
    for (uint32_t h = 0; h < n_qh; ++h) {
        for (uint32_t i = 0; i < seq; ++i) {
            float max_s = -1e30f;
            for (uint32_t j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dot += q[(h*seq+i)*dim+d] * k[k_off(j)+d];
                max_s = (std::max)(max_s, dot * scale);
            }
            float sum = 0.0f;
            std::vector<float> w(seq);
            for (uint32_t j = 0; j < seq; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dot += q[(h*seq+i)*dim+d] * k[k_off(j)+d];
                w[j] = std::exp(dot * scale - max_s);
                sum += w[j];
            }
            float ref = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < seq; ++j) acc += (w[j]/sum) * v[v_off(j)+d];
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
    printf("\nResult: %s\n", ok ? "PagedAttention PASSED" : "FAILED");
    return ok ? 0 : 1;
}
