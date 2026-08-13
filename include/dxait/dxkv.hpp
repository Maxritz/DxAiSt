#ifndef DXAIT_DXKV_HPP
#define DXAIT_DXKV_HPP

#include "dxait.hpp"
#include <vector>

namespace dxait {

struct KVCacheConfig {
    uint32_t num_layers{32};
    uint32_t num_heads{32};
    uint32_t head_dim{128};
    uint32_t max_seq_len{4096};
    uint32_t page_size{16};
};

class KVCacheManager {
public:
    KVCacheManager(Device* device, const KVCacheConfig& config);
    ~KVCacheManager() = default;

    uint32_t allocate_sequence();
    void free_sequence(uint32_t seq_id);
    Buffer* get_key_buffer() const { return m_key_buffer.get(); }
    Buffer* get_value_buffer() const { return m_val_buffer.get(); }

private:
    Device* m_device;
    KVCacheConfig m_config;
    std::unique_ptr<Buffer> m_key_buffer;
    std::unique_ptr<Buffer> m_val_buffer;
    std::vector<bool> m_allocated_pages;
};

} // namespace dxait

#endif // DXAIT_DXKV_HPP
