#ifndef DXAIT_DXCACHE_HPP
#define DXAIT_DXCACHE_HPP

#include "dxait.hpp"
#include <vector>
#include <memory>

namespace dxait {

enum class KVCacheType {
    Standard = 0,
    RadixTree,
    HadamardTransform
};

class AdvancedKVCache {
public:
    AdvancedKVCache(Device* device, KVCacheType type, uint64_t max_bytes);
    ~AdvancedKVCache() = default;

    KVCacheType type() const { return m_type; }
    Buffer* get_buffer() const { return m_cache_buffer.get(); }
    void apply_hadamard_transform(Queue* queue);

private:
    Device* m_device;
    KVCacheType m_type;
    std::unique_ptr<Buffer> m_cache_buffer;
};

} // namespace dxait

#endif // DXAIT_DXCACHE_HPP
