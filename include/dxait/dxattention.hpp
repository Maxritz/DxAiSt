#ifndef DXAIT_DXATTENTION_HPP
#define DXAIT_DXATTENTION_HPP

#include "dxait.hpp"
#include "dxjit.hpp"
#include <string>
#include <vector>

namespace dxait {

enum class AttentionMechanism {
    MHA,               // Multi-Head Attention (Vaswani et al., 2017)
    GQA,               // Grouped-Query Attention (Ainslie et al., 2023)
    MQA,               // Multi-Query Attention (Shazeer, 2019)
    FlashAttention,    // FlashAttention-2 / 3 Tiled Online Softmax (Dao et al., 2022)
    SlidingWindow,     // Sliding Window Attention (Beltagy et al., 2020 / Mistral AI)
    PagedAttention,    // PagedAttention Virtual Memory KV Cache (Kwon et al., 2023)
    ChunkedPrefill,    // Chunked Prefill & Decode Piggybacking (Agrawal et al., 2023)
    RingAttention,     // Distributed Sequence Parallel Ring Attention (Liu et al., 2023)
    HeavyHitterH2O,    // KV Cache Compression & Heavy Hitter Eviction (Zhang et al., 2023)
    LinearAttention    // Recurrent State & Linear Attention (Katharopoulos et al., 2020 / Mamba)
};

struct AttentionConfig {
    AttentionMechanism mechanism{AttentionMechanism::FlashAttention};
    uint32_t num_q_heads{32};
    uint32_t num_kv_heads{8};       // GQA 4:1 ratio (or 1 for MQA)
    uint32_t head_dim{128};
    uint32_t seq_len{4096};
    uint32_t sliding_window_size{4096};
    uint32_t page_block_size{16};   // PagedAttention block size
    float scale{0.088388347f};      // 1 / sqrt(head_dim)
    bool use_quantized_kv{true};    // Q4_0 / Q8_0 KV Cache
};

class AttentionOps {
public:
    explicit AttentionOps(Device* device);
    ~AttentionOps() = default;

    // 1. Basic Scaled Dot-Product Attention
    void scaled_dot_product_attention(
        Queue* queue,
        Buffer* out_buf,
        Buffer* q_buf,
        Buffer* k_buf,
        Buffer* v_buf,
        uint32_t seq_len,
        uint32_t head_dim,
        float scale
    );

    // 2. Comprehensive Master Attention Dispatcher (MHA, GQA, MQA, FlashAttn, PagedAttn, SWA, RingAttn, H2O, ChunkedPrefill)
    // extra = block table (PagedAttention) or keep mask (H2O); nullptr otherwise.
    void dispatch_attention(
        Queue* queue,
        Buffer* out_buf,
        Buffer* q_buf,
        Buffer* k_buf,
        Buffer* v_buf,
        const AttentionConfig& config,
        Buffer* extra = nullptr
    );

private:
    Device* m_device;
    PipelineCache m_pso_cache;
    ComPtr<ID3D12RootSignature> m_root_sig;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;
    std::unique_ptr<Fence> m_fence;
    uint64_t m_fence_val{0};

    void init_root_signature();
};

} // namespace dxait

#endif // DXAIT_DXATTENTION_HPP
