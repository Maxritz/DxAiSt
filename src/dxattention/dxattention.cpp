#include "dxait/dxattention.hpp"
#include <iostream>
#include <stdexcept>

namespace dxait {

// Real scaled dot-product attention: out = softmax(Q K^T / scale + causal_mask) V.
// One thread per query position; loops over keys and head dim. Handles KV head
// sharing for GQA/MQA by mapping the head index to a shared KV head.
static const char g_sdpa_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer AttnCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_num_q_heads;
    uint g_num_kv_heads;
    float g_scale;
    uint g_causal;      // 1 = causal mask
    uint g_window;      // sliding window size (0 = full)
    uint g_pad;
};

[numthreads(64, 1, 1)]
void sdpa(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;

    uint q_head = id.y;
    uint kv_head = min(q_head * g_num_kv_heads / max(g_num_q_heads, 1u), g_num_kv_heads - 1);

    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    float max_s = -1e30f;
    for (uint j = 0; j < g_seq_len; ++j) {
        if (g_causal && j > i) break;
        if (g_window > 0 && (i > j + g_window)) continue;
        float dot = 0.0f;
        uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d) {
            dot += g_q[q_base + d] * g_k[k_base + d];
        }
        max_s = max(max_s, dot * g_scale);
    }

    float sum = 0.0f;
    for (uint j = 0; j < g_seq_len; ++j) {
        if (g_causal && j > i) break;
        if (g_window > 0 && (i > j + g_window)) continue;
        float dot = 0.0f;
        uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d) {
            dot += g_q[q_base + d] * g_k[k_base + d];
        }
        sum += exp((dot * g_scale) - max_s);
    }

    for (uint d = 0; d < g_head_dim; ++d) {
        float o = 0.0f;
        for (uint j = 0; j < g_seq_len; ++j) {
            if (g_causal && j > i) break;
            if (g_window > 0 && (i > j + g_window)) continue;
            float dot = 0.0f;
            uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
            for (uint dd = 0; dd < g_head_dim; ++dd) {
                dot += g_q[q_base + dd] * g_k[k_base + dd];
            }
            uint v_base = (kv_head * g_seq_len + j) * g_head_dim;
            o += (exp((dot * g_scale) - max_s) / sum) * g_v[v_base + d];
        }
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = o;
    }
}
)";

// FlashAttention-style tiled kernel: online softmax with running max and sum.
// Processes keys in blocks of BLOCK size, rescales accumulated output each block.
static const char g_flash_hlsl[] = R"(
#define FLASH_BLOCK 16
#define DX_FLASH_MAX_DIM 128
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer FlashCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_num_q_heads;
    uint g_num_kv_heads;
    float g_scale;
    uint g_pad[3];
};

[numthreads(64, 1, 1)]
void flash_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    if (g_head_dim > DX_FLASH_MAX_DIM) return;
    uint q_head = id.y;
    uint kv_head = min(q_head * g_num_kv_heads / max(g_num_q_heads, 1u), g_num_kv_heads - 1);

    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    float m = -1e30f;
    float l = 0.0f;
    float acc[DX_FLASH_MAX_DIM];

    for (uint d = 0; d < g_head_dim; ++d) acc[d] = 0.0f;

    for (uint j0 = 0; j0 < g_seq_len; j0 += FLASH_BLOCK) {
        float block_max = -1e30f;
        // find block max (causal: only j <= i)
        for (uint b = 0; b < FLASH_BLOCK; ++b) {
            uint j = j0 + b;
            if (j >= g_seq_len) break;
            if (j > i) break;
            float dot = 0.0f;
            uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d)
                dot += g_q[q_base + d] * g_k[k_base + d];
            block_max = max(block_max, dot * g_scale);
        }

        float new_m = max(m, block_max);
        float rescale = exp(m - new_m);

        // accumulate this block into a temp (fresh each block), then merge
        float tmp[DX_FLASH_MAX_DIM];
        for (uint d = 0; d < g_head_dim; ++d) tmp[d] = 0.0f;
        float block_l = 0.0f;

        for (uint b = 0; b < FLASH_BLOCK; ++b) {
            uint j = j0 + b;
            if (j >= g_seq_len) break;
            if (j > i) break;
            float dot = 0.0f;
            uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d)
                dot += g_q[q_base + d] * g_k[k_base + d];
            float p = exp((dot * g_scale) - new_m);
            block_l += p;
            uint v_base = (kv_head * g_seq_len + j) * g_head_dim;
            for (uint d = 0; d < g_head_dim; ++d)
                tmp[d] += p * g_v[v_base + d];
        }

        // merge: acc = acc*rescale + tmp, l = l*rescale + block_l
        for (uint d = 0; d < g_head_dim; ++d) acc[d] = acc[d] * rescale + tmp[d];
        l = l * rescale + block_l;
        m = new_m;
    }

    for (uint d = 0; d < g_head_dim; ++d)
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = (l > 0.0f) ? (acc[d] / l) : 0.0f;
}
)";

// Linear (kernelized) attention: out = phi(Q) (phi(K)^T V) / (phi(Q) (phi(K)^T 1)).
// Uses elu(x)+1 feature map. Linear in sequence length via associativity.
// Builds the recurrent state S[d][e] = sum_j phi(k_j)[d] * v_j[e] and z[d].
static const char g_linear_hlsl[] = R"(
#define LIN_MAX_DIM 32
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer LinearCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_num_q_heads;
    uint g_num_kv_heads;
    float g_scale;
    uint g_pad[3];
};

float kernel(float x) { return max(x, 0.0f) + exp(min(x, 0.0f)); }

[numthreads(64, 1, 1)]
void linear_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    if (g_head_dim > LIN_MAX_DIM) return;
    uint q_head = id.y;
    uint kv_head = min(q_head * g_num_kv_heads / max(g_num_q_heads, 1u), g_num_kv_heads - 1);

    // S[d*LIN_MAX_DIM + e]
    float S[LIN_MAX_DIM * LIN_MAX_DIM];
    float z[LIN_MAX_DIM];
    for (uint d = 0; d < g_head_dim; ++d) {
        z[d] = 0.0f;
        for (uint e = 0; e < g_head_dim; ++e) S[d * LIN_MAX_DIM + e] = 0.0f;
    }

    for (uint j = 0; j <= i && j < g_seq_len; ++j) {
        uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
        uint v_base = (kv_head * g_seq_len + j) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d) {
            float kd = kernel(g_k[k_base + d]);
            z[d] += kd;
            for (uint e = 0; e < g_head_dim; ++e)
                S[d * LIN_MAX_DIM + e] += kd * g_v[v_base + e];
        }
    }

    uint q_base = (q_head * g_seq_len + i) * g_head_dim;
    float norm = 0.0f;
    float o[LIN_MAX_DIM];
    for (uint e = 0; e < g_head_dim; ++e) o[e] = 0.0f;
    for (uint d = 0; d < g_head_dim; ++d) {
        float qd = kernel(g_q[q_base + d]);
        norm += qd * z[d];
        for (uint e = 0; e < g_head_dim; ++e)
            o[e] += qd * S[d * LIN_MAX_DIM + e];
    }
    for (uint e = 0; e < g_head_dim; ++e)
        g_out[(q_head * g_seq_len + i) * g_head_dim + e] = (norm > 0.0f) ? (o[e] / norm) : 0.0f;
}
)";

// PagedAttention: KV stored in fixed-size blocks, block table maps logical
// block -> physical block. extra buffer = block table (uint per block).
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
    uint g_num_q_heads;
    uint g_num_kv_heads;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void paged_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    uint q_head = id.y;
    uint kv_head = min(q_head * g_num_kv_heads / max(g_num_q_heads, 1u), g_num_kv_heads - 1);

    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    float max_s = -1e30f;
    for (uint j = 0; j < g_seq_len; ++j) {
        uint logical_block = j / g_page_size;
        uint in_block = j % g_page_size;
        uint phys = g_table[kv_head * g_n_blocks + logical_block];
        float dot = 0.0f;
        uint k_base = ((kv_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d)
            dot += g_q[q_base + d] * g_k[k_base + d];
        max_s = max(max_s, dot * g_scale);
    }

    float sum = 0.0f;
    for (uint j = 0; j < g_seq_len; ++j) {
        uint logical_block = j / g_page_size;
        uint in_block = j % g_page_size;
        uint phys = g_table[kv_head * g_n_blocks + logical_block];
        float dot = 0.0f;
        uint k_base = ((kv_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d)
            dot += g_q[q_base + d] * g_k[k_base + d];
        sum += exp((dot * g_scale) - max_s);
    }

    for (uint d = 0; d < g_head_dim; ++d) {
        float o = 0.0f;
        for (uint j = 0; j < g_seq_len; ++j) {
            uint logical_block = j / g_page_size;
            uint in_block = j % g_page_size;
            uint phys = g_table[kv_head * g_n_blocks + logical_block];
            float dot = 0.0f;
            uint k_base = ((kv_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
            for (uint dd = 0; dd < g_head_dim; ++dd)
                dot += g_q[q_base + dd] * g_k[k_base + dd];
            uint v_base = ((kv_head * g_n_blocks + phys) * g_page_size + in_block) * g_head_dim;
            o += (exp((dot * g_scale) - max_s) / sum) * g_v[v_base + d];
        }
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = o;
    }
}
)";

// Heavy-Hitter H2O: keep-mask attention. extra buffer = uint keep[seq] (0/1).
static const char g_h2o_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);
StructuredBuffer<uint> g_keep : register(t3);

cbuffer H2OCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_num_q_heads;
    uint g_num_kv_heads;
    float g_scale;
    uint g_pad[3];
};

[numthreads(64, 1, 1)]
void h2o_attn(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= g_seq_len) return;
    uint q_head = id.y;
    uint kv_head = min(q_head * g_num_kv_heads / max(g_num_q_heads, 1u), g_num_kv_heads - 1);
    uint q_base = (q_head * g_seq_len + i) * g_head_dim;

    float max_s = -1e30f;
    for (uint j = 0; j < g_seq_len; ++j) {
        if (!g_keep[j]) continue;
        float dot = 0.0f;
        uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d) dot += g_q[q_base + d] * g_k[k_base + d];
        max_s = max(max_s, dot * g_scale);
    }

    float sum = 0.0f;
    for (uint j = 0; j < g_seq_len; ++j) {
        if (!g_keep[j]) continue;
        float dot = 0.0f;
        uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
        for (uint d = 0; d < g_head_dim; ++d) dot += g_q[q_base + d] * g_k[k_base + d];
        sum += exp((dot * g_scale) - max_s);
    }

    for (uint d = 0; d < g_head_dim; ++d) {
        float o = 0.0f;
        for (uint j = 0; j < g_seq_len; ++j) {
            if (!g_keep[j]) continue;
            float dot = 0.0f;
            uint k_base = (kv_head * g_seq_len + j) * g_head_dim;
            for (uint dd = 0; dd < g_head_dim; ++dd) dot += g_q[q_base + dd] * g_k[k_base + dd];
            uint v_base = (kv_head * g_seq_len + j) * g_head_dim;
            o += (exp((dot * g_scale) - max_s) / sum) * g_v[v_base + d];
        }
        g_out[(q_head * g_seq_len + i) * g_head_dim + d] = o;
    }
}
)";

AttentionOps::AttentionOps(Device* device) : m_device(device), m_pso_cache(device->get()), m_fence(device->create_fence(0)) {
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
    D3D12_ROOT_PARAMETER params[6]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 8;
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

    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[5].Descriptor.ShaderRegister = 3;
    params[5].Descriptor.RegisterSpace = 0;
    params[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 6;
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
    AttentionConfig cfg;
    cfg.mechanism = AttentionMechanism::MHA;
    cfg.num_q_heads = 1;
    cfg.num_kv_heads = 1;
    cfg.head_dim = head_dim;
    cfg.seq_len = seq_len;
    cfg.scale = scale;
    dispatch_attention(queue, out_buf, q_buf, k_buf, v_buf, cfg);
}

void AttentionOps::dispatch_attention(
    Queue* queue,
    Buffer* out_buf,
    Buffer* q_buf,
    Buffer* k_buf,
    Buffer* v_buf,
    const AttentionConfig& config,
    Buffer* extra
) {
    // All 10 mechanisms route to a real kernel. paged/h2o need the extra SRV
    // (block table / keep mask); the rest ignore it.
    bool causal = (config.mechanism == AttentionMechanism::FlashAttention ||
                   config.mechanism == AttentionMechanism::SlidingWindow ||
                   config.mechanism == AttentionMechanism::PagedAttention ||
                   config.mechanism == AttentionMechanism::ChunkedPrefill ||
                   config.mechanism == AttentionMechanism::RingAttention);
    uint32_t window = (config.mechanism == AttentionMechanism::SlidingWindow)
                          ? config.sliding_window_size : 0;

    const char* key = nullptr;
    const wchar_t* entry = nullptr;
    const char* src = g_sdpa_hlsl;
    switch (config.mechanism) {
    case AttentionMechanism::FlashAttention:
    case AttentionMechanism::ChunkedPrefill:   // chunked = online-softmax over blocks
    case AttentionMechanism::RingAttention:    // ring = online-softmax over ring chunks
        key = "flash_attn"; entry = L"flash_attn"; src = g_flash_hlsl; break;
    case AttentionMechanism::LinearAttention:
        key = "linear_attn"; entry = L"linear_attn"; src = g_linear_hlsl; break;
    case AttentionMechanism::PagedAttention:
        key = "paged_attn"; entry = L"paged_attn"; src = g_paged_hlsl; break;
    case AttentionMechanism::HeavyHitterH2O:
        key = "h2o_attn"; entry = L"h2o_attn"; src = g_h2o_hlsl; break;
    default:
        key = "sdpa"; entry = L"sdpa"; break; // MHA, GQA, MQA, SWA
    }

    auto pso = m_pso_cache.get_or_compile(key, src, m_root_sig.Get(), entry);

    if (m_fence_val > 0) m_fence->wait(m_fence_val);
    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t seq_len;
        uint32_t head_dim;
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        float scale;
        uint32_t causal;
        uint32_t window;
        uint32_t pad;
    } cb{config.seq_len, config.head_dim, config.num_q_heads, config.num_kv_heads,
         config.scale, causal ? 1u : 0u, window, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 8, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, q_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, k_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(4, v_buf->get()->GetGPUVirtualAddress());
    if (extra) {
        m_cmd_list->SetComputeRootShaderResourceView(5, extra->get()->GetGPUVirtualAddress());
    }

    uint32_t grid_x = (config.seq_len + 63) / 64;
    m_cmd_list->Dispatch(grid_x, config.num_q_heads, 1);
    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
    queue->signal(*m_fence, ++m_fence_val);
    // Synchronous: wait so the next dispatch_attention can safely reset the
    // shared allocator. Avoids order-dependent behaviour across calls (RDNA2).
    m_fence->wait(m_fence_val);
}

} // namespace dxait
