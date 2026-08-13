#ifndef DXAIT_DXCONTEXT_HPP
#define DXAIT_DXCONTEXT_HPP

#include "dxait.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dxait {

enum class ContextQuantType {
    None,     // FP16 / FP32 uncompressed
    Int8,     // 8-bit linear quantization (2x RAM savings)
    Q4_0      // 4-bit block quantization (4x RAM savings)
};

struct ContextConfig {
    uint32_t max_tokens{524288};       // 512k token maximum capacity
    uint32_t active_target_tokens{131072}; // Dynamically requested context window
    uint32_t num_heads{32};
    uint32_t head_dim{128};
    uint32_t sliding_window{4096};     // VRAM hot window size
    ContextQuantType quant_type{ContextQuantType::Q4_0};
    bool enable_offloading{true};
    bool enable_compression{true};
};

struct ContextStats {
    uint32_t active_tokens{0};
    uint32_t target_tokens{0};
    uint32_t vram_tokens{0};
    uint32_t sysram_tokens{0};
    uint64_t vram_bytes{0};
    uint64_t sysram_bytes{0};
    float compression_ratio{1.0f};
};

class LongContextEngine {
public:
    LongContextEngine(Device* device, const ContextConfig& config);
    ~LongContextEngine();

    // 1. Dynamic User Context Target Configuration
    void set_target_context_length(uint32_t target_tokens);

    // 2. Context KV Cache Ingestion
    void append_tokens(uint32_t num_tokens);

    // 3. Translate / Compress Large Context to Fit Smaller Model Window (e.g., 512K -> 4K)
    void translate_context_for_model(uint32_t model_max_context, Buffer* source_kv, Buffer* dest_kv);

    // 4. Quantization, Offloading & Compression
    void quantize_kv_cache(Buffer* kv_in, Buffer* kv_out, uint32_t token_count, ContextQuantType type);
    void offload_cold_context(Queue* copy_queue);
    void prefetch_context(Queue* copy_queue, uint32_t start_token, uint32_t count);
    void compress_context_tokens(Buffer* kv_buf, uint32_t current_tokens, float ratio);

    ContextStats get_stats() const;

private:
    Device* m_device;
    ContextConfig m_config;
    ContextStats m_stats{};

    std::unique_ptr<Buffer> m_vram_kv_cache;
    std::unique_ptr<Buffer> m_sysram_kv_cache;

    // ponytail: anchor + stride downsampling for context translation; upgrade path is KV activation pooling
};

} // namespace dxait

#endif // DXAIT_DXCONTEXT_HPP
