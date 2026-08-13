#include "dxait/dxkv.hpp"

namespace dxait {

KVCacheManager::KVCacheManager(Device* device, const KVCacheConfig& config)
    : m_device(device), m_config(config) {
    uint64_t total_elements = static_cast<uint64_t>(config.num_layers) * config.num_heads * config.max_seq_len * config.head_dim;
    uint64_t bytes_per_buffer = total_elements * sizeof(uint16_t); // FP16 KV cache

    m_key_buffer = device->create_buffer(bytes_per_buffer, MemLocation::Default);
    m_val_buffer = device->create_buffer(bytes_per_buffer, MemLocation::Default);

    uint32_t total_pages = (config.max_seq_len + config.page_size - 1) / config.page_size;
    m_allocated_pages.resize(total_pages, false);
}

uint32_t KVCacheManager::allocate_sequence() {
    for (size_t i = 0; i < m_allocated_pages.size(); ++i) {
        if (!m_allocated_pages[i]) {
            m_allocated_pages[i] = true;
            return static_cast<uint32_t>(i);
        }
    }
    return 0xFFFFFFFF; // Pool exhausted
}

void KVCacheManager::free_sequence(uint32_t seq_id) {
    if (seq_id < m_allocated_pages.size()) {
        m_allocated_pages[seq_id] = false;
    }
}

} // namespace dxait
