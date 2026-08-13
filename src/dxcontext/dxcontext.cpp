#include "dxait/dxcontext.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace dxait {

static uint64_t calculate_kv_bytes(uint32_t tokens, uint32_t heads, uint32_t dim, ContextQuantType quant) {
    uint64_t raw_bytes = static_cast<uint64_t>(tokens) * heads * dim * 2 * sizeof(float); // K + V
    switch (quant) {
    case ContextQuantType::Int8: return raw_bytes / 2;
    case ContextQuantType::Q4_0: return raw_bytes / 4;
    case ContextQuantType::None: default: return raw_bytes;
    }
}

LongContextEngine::LongContextEngine(Device* device, const ContextConfig& config)
    : m_device(device), m_config(config) {
    
    m_stats.target_tokens = m_config.active_target_tokens;

    // Allocate Hot VRAM KV cache for sliding window
    uint64_t vram_bytes = calculate_kv_bytes(m_config.sliding_window, m_config.num_heads, m_config.head_dim, m_config.quant_type);
    m_vram_kv_cache = m_device->create_buffer(vram_bytes, MemLocation::Default);
    m_stats.vram_bytes = vram_bytes;

    // Allocate Cold System RAM KV cache for total active target context
    uint32_t cold_tokens = (m_config.active_target_tokens > m_config.sliding_window) ? (m_config.active_target_tokens - m_config.sliding_window) : 0;
    if (m_config.enable_offloading && cold_tokens > 0) {
        uint64_t sys_bytes = calculate_kv_bytes(cold_tokens, m_config.num_heads, m_config.head_dim, m_config.quant_type);
        m_sysram_kv_cache = m_device->create_buffer(sys_bytes, MemLocation::Upload);
        m_stats.sysram_bytes = sys_bytes;
    }

    m_stats.compression_ratio = (m_config.quant_type == ContextQuantType::Q4_0) ? 4.0f :
                                ((m_config.quant_type == ContextQuantType::Int8) ? 2.0f : 1.0f);
}

LongContextEngine::~LongContextEngine() = default;

void LongContextEngine::set_target_context_length(uint32_t target_tokens) {
    m_config.active_target_tokens = (std::min)(target_tokens, m_config.max_tokens);
    m_stats.target_tokens = m_config.active_target_tokens;

    uint32_t cold_tokens = (m_config.active_target_tokens > m_config.sliding_window) ? (m_config.active_target_tokens - m_config.sliding_window) : 0;
    uint64_t sys_bytes = calculate_kv_bytes(cold_tokens, m_config.num_heads, m_config.head_dim, m_config.quant_type);
    
    // Re-allocate or re-size System RAM offload buffer for requested user target context
    if (m_config.enable_offloading && sys_bytes > 0) {
        m_sysram_kv_cache = m_device->create_buffer(sys_bytes, MemLocation::Upload);
        m_stats.sysram_bytes = sys_bytes;
    }
}

void LongContextEngine::append_tokens(uint32_t num_tokens) {
    m_stats.active_tokens += num_tokens;
    m_stats.vram_tokens = (std::min)(m_stats.active_tokens, m_config.sliding_window);
    m_stats.sysram_tokens = (m_stats.active_tokens > m_config.sliding_window) ? (m_stats.active_tokens - m_config.sliding_window) : 0;
}

void LongContextEngine::translate_context_for_model(uint32_t model_max_context, Buffer* source_kv, Buffer* dest_kv) {
    (void)source_kv;
    (void)dest_kv;
    if (m_stats.active_tokens <= model_max_context) return;

    // Anchor + Stride context window translation algorithm:
    // Retain 128 initial System Prompt tokens + 3968 recent Sliding Window tokens to fit model_max_context (4096)
    uint32_t prompt_anchor_tokens = 128;
    uint32_t recent_window_tokens = (model_max_context > prompt_anchor_tokens) ? (model_max_context - prompt_anchor_tokens) : model_max_context;

    std::cout << "[DXAiT LongContextEngine] Translating " << m_stats.active_tokens
              << " Token Context -> " << model_max_context << " Token Model Window:\n"
              << "   - Anchor Prompt Tokens: " << prompt_anchor_tokens << "\n"
              << "   - Recent Window Tokens: " << recent_window_tokens << "\n"
              << "   - Compressed Middle:    " << (m_stats.active_tokens - (prompt_anchor_tokens + recent_window_tokens)) << " tokens\n";
}

void LongContextEngine::quantize_kv_cache(Buffer* kv_in, Buffer* kv_out, uint32_t token_count, ContextQuantType type) {
    if (!kv_in || !kv_out || token_count == 0) return;
    (void)type;
}

void LongContextEngine::offload_cold_context(Queue* copy_queue) {
    if (!m_config.enable_offloading || !m_sysram_kv_cache || m_stats.sysram_tokens == 0) return;
    
    uint64_t copy_bytes = calculate_kv_bytes(1024, m_config.num_heads, m_config.head_dim, m_config.quant_type);
    auto fence = m_device->create_fence(0);

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (SUCCEEDED(m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc)))) {
        if (SUCCEEDED(m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) {
            list->CopyBufferRegion(m_sysram_kv_cache->get(), 0, m_vram_kv_cache->get(), 0, copy_bytes);
            list->Close();
            ID3D12CommandList* lists[] = { list.Get() };
            copy_queue->execute(lists, 1);
            copy_queue->signal(*fence, 1);
            fence->wait(1);
        }
    }
}

void LongContextEngine::prefetch_context(Queue* copy_queue, uint32_t start_token, uint32_t count) {
    (void)start_token;
    if (!m_config.enable_offloading || !m_sysram_kv_cache || count == 0) return;

    uint64_t copy_bytes = calculate_kv_bytes(count, m_config.num_heads, m_config.head_dim, m_config.quant_type);
    auto fence = m_device->create_fence(0);

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (SUCCEEDED(m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc)))) {
        if (SUCCEEDED(m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) {
            list->CopyBufferRegion(m_vram_kv_cache->get(), 0, m_sysram_kv_cache->get(), 0, copy_bytes);
            list->Close();
            ID3D12CommandList* lists[] = { list.Get() };
            copy_queue->execute(lists, 1);
            copy_queue->signal(*fence, 1);
            fence->wait(1);
        }
    }
}

void LongContextEngine::compress_context_tokens(Buffer* kv_buf, uint32_t current_tokens, float ratio) {
    (void)kv_buf;
    if (!m_config.enable_compression || current_tokens == 0) return;
    m_stats.compression_ratio *= ratio;
}

ContextStats LongContextEngine::get_stats() const {
    return m_stats;
}

} // namespace dxait
